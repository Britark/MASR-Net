# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp  # 添加混合精度支持
import kornia.losses  # 添加kornia损失函数

import feature_extractor
from feature_extractor import FeatureExtractor
import config
from config import default_config
import emb_gen
from MoA import MultiheadAttention, MoE
from utils import *
from decoder import *
from ISP import *
from losses import *
from torchvision.models import vgg16


# ==================== 反棋盘格模块 ====================

class MultiScaleParamSmoother(nn.Module):
    """
    多尺度参数平滑模块
    使用5x5、7x7卷积进行多尺度平滑
    使用reflect padding避免边缘黑边伪影
    """
    def __init__(self, in_channels, residual_weight=0.1):
        super().__init__()
        self.residual_weight = nn.Parameter(torch.tensor(residual_weight))

        # 2个不同尺度的卷积，使用reflect padding
        self.conv5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, padding_mode='reflect', bias=False)
        self.conv7 = nn.Conv2d(in_channels, in_channels, 7, padding=3, padding_mode='reflect', bias=False)

        # 融合层
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1, bias=True)

        # 初始化为平滑滤波器
        with torch.no_grad():
            for conv in [self.conv5, self.conv7]:
                nn.init.constant_(conv.weight, 1.0 / (conv.kernel_size[0] ** 2))
            nn.init.xavier_uniform_(self.fusion.weight, gain=0.1)
            nn.init.zeros_(self.fusion.bias)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 参数特征图
        Returns:
            平滑后的特征图 [B, C, H, W]
        """
        identity = x

        # 多尺度卷积
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        # 拼接并融合
        x_cat = torch.cat([x5, x7], dim=1)
        x_fused = self.fusion(x_cat)

        # 残差连接
        return identity + self.residual_weight * x_fused


class EnhancedFFN(nn.Module):
    """
    增强的前馈网络，替换原简单MLP
    """
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # 保守初始化
        with torch.no_grad():
            nn.init.xavier_uniform_(self.net[0].weight, gain=0.3)
            nn.init.zeros_(self.net[0].bias)
            nn.init.xavier_uniform_(self.net[3].weight, gain=0.3)
            nn.init.zeros_(self.net[3].bias)

    def forward(self, x):
        """
        Args:
            x: [B, H, W, C] 参数特征
        Returns:
            增强后的特征 [B, H, W, C]
        """
        return x + self.residual_weight * self.net(x)


class AntiCheckerboardModule(nn.Module):
    """
    图像空间反棋盘格模块
    在像素级参数图上应用反棋盘格平滑
    使用reflect padding避免边缘黑边伪影
    """
    def __init__(self, channels, residual_weight=0.1):
        super().__init__()
        self.residual_weight = nn.Parameter(torch.tensor(residual_weight))

        # 3个分支：横向、纵向、对角线，使用reflect padding
        self.horizontal = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), padding_mode='reflect', bias=False)
        self.vertical = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), padding_mode='reflect', bias=False)
        self.diagonal = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False)

        # 融合层
        self.fusion = nn.Conv2d(channels * 3, channels, 1, bias=True)

        # 初始化
        with torch.no_grad():
            for conv in [self.horizontal, self.vertical, self.diagonal]:
                nn.init.constant_(conv.weight, 1.0 / conv.weight.numel() * channels)
            nn.init.xavier_uniform_(self.fusion.weight, gain=0.1)
            nn.init.zeros_(self.fusion.bias)

    def forward(self, x):
        """
        Args:
            x: [B, H, W, C] 图像空间参数
        Returns:
            反棋盘格处理后 [B, H, W, C]
        """
        # 转换为 [B, C, H, W]
        x_t = x.permute(0, 3, 1, 2)
        identity = x_t

        # 3个方向的平滑
        h = self.horizontal(x_t)
        v = self.vertical(x_t)
        d = self.diagonal(x_t)

        # 融合
        fused = self.fusion(torch.cat([h, v, d], dim=1))

        # 残差连接
        out = identity + self.residual_weight * fused

        # 转换回 [B, H, W, C]
        return out.permute(0, 2, 3, 1)


class BoundaryAwareSmoothing(nn.Module):
    """
    边界感知平滑模块
    根据图像边缘信息自适应调整平滑强度
    使用reflect padding避免边缘黑边伪影
    """
    def __init__(self, in_channels, residual_weight=0.1):
        super().__init__()
        self.residual_weight = nn.Parameter(torch.tensor(residual_weight))

        # 边缘检测，使用reflect padding
        self.edge_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect', bias=False)

        # 平滑卷积，使用reflect padding
        self.smooth_conv = nn.Conv2d(in_channels, in_channels, 5, padding=2, padding_mode='reflect', bias=False)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

        # 初始化
        with torch.no_grad():
            # Sobel-like初始化用于边缘检测
            nn.init.xavier_uniform_(self.edge_conv.weight, gain=1.0)
            # 平滑滤波器初始化
            nn.init.constant_(self.smooth_conv.weight, 1.0 / 25.0)
            # 门控初始化
            nn.init.xavier_uniform_(self.gate[0].weight, gain=0.1)
            nn.init.constant_(self.gate[0].bias, 0.5)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 参数特征图
        Returns:
            边界感知平滑后 [B, C, H, W]
        """
        identity = x

        # 边缘响应
        edge = self.edge_conv(x)

        # 平滑
        smooth = self.smooth_conv(x)

        # 自适应门控
        gate_input = torch.cat([edge, smooth], dim=1)
        gate_value = self.gate(gate_input)

        # 融合
        out = gate_value * smooth + (1 - gate_value) * identity

        # 残差连接
        return identity + self.residual_weight * (out - identity)


# ==================== 主模型 ====================

class Model(nn.Module):
    """主模型类，组装各个预定义的模块"""

    def __init__(self, config=None):
        """
        使用配置字典初始化模型
        Args:
            config: 包含所有模型参数的配置字典，如果为None则使用默认配置
        """
        super(Model, self).__init__()
        self.expert_indices = None
        # 如果没有提供配置，使用默认配置
        if config is None:
            config = default_config()

        # 从config读取图像尺寸参数，设置H和W
        self.patch_size = config['image_size']['patch_size']
        self.win_size = config['image_size']['win_size']

        # 添加损失权重超参数
        loss_weights_config = config.get('loss_weights', {})
        self.reconstruction_weight = loss_weights_config.get('reconstruction_weight', 1.0)  # 重建损失权重
        self.auxiliary_weight = loss_weights_config.get('auxiliary_weight', 0.1)  # 辅助损失权重
        self.perceptual_weight = loss_weights_config.get('perceptual_weight', 0.1)
        self.psnr_weight = loss_weights_config.get('psnr_weight', 0.28)  # PSNR损失权重 (降低)
        self.ssim_weight = loss_weights_config.get('ssim_weight', 0.3)  # SSIM损失权重 (降低)
        self.lab_color_weight = loss_weights_config.get('lab_color_weight', 0.02)  # LAB色彩损失权重

        # 获取特征提取器配置
        fe_config = config.get('feature_extractor', {})

        # 实例化特征提取器
        self.feature_extractor = FeatureExtractor(
            patch_size=fe_config.get('patch_size', 2),  # patch_size=4 win_size=5&& patch_size=8 win_size=7
            in_channels=fe_config.get('in_channels', 3),
            embed_dim=fe_config.get('embed_dim', 128),
            win_size=fe_config.get('win_size', 5)
        )

        # 获取EispGeneratorFFN配置
        emb_gen_config = config.get('emb_gen', {})

        # 获取QKIspGenerator配置（提前获取以使用num_heads）
        qk_gen_config = config.get('qk_gen', {})

        # 实例化EispGeneratorFFN
        self.emb_gen = emb_gen.EispGeneratorFFN(
            input_dim=emb_gen_config.get('input_dim', 128),
            hidden_dim=emb_gen_config.get('hidden_dim', 512),
            output_dim=emb_gen_config.get('output_dim', 128),
            dropout_rate=emb_gen_config.get('dropout_rate', 0.1)
        )

        # 添加线性层，将e_isp的最后一个维度映射为其num_heads倍大
        self.e_isp_expand = nn.Linear(
            emb_gen_config.get('output_dim', 128),
            emb_gen_config.get('output_dim', 128) * qk_gen_config.get('num_heads', 3)
        )

        # 实例化QKIspGenerator
        self.qk_gen = emb_gen.QKIspGenerator(
            dim=qk_gen_config.get('dim', 128),
            num_heads=qk_gen_config.get('num_heads', 3),
            dropout_rate=qk_gen_config.get('dropout_rate', 0.1)
        )

        # 获取KVFeatureGenerator配置
        kv_gen_config = config.get('kv_gen', {})

        # 实例化KVFeatureGenerator
        self.kv_gen = emb_gen.KVFeatureGenerator(
            embed_dim=kv_gen_config.get('embed_dim', 128),
            dropout_rate=kv_gen_config.get('dropout_rate', 0.1)
        )

        # 获取MultiheadAttention配置
        multi_attn_1_config = config.get('multi_attention_1', {})

        # 实例化MultiheadAttention
        embed_dim_1 = multi_attn_1_config.get('embed_dim', 128)
        num_heads_1 = multi_attn_1_config.get('num_heads', 2)
        self.multi_attention_1 = MultiheadAttention(
            embed_dim=embed_dim_1,
            num_heads=num_heads_1,
            dropout=multi_attn_1_config.get('dropout', 0.1),
            bias=multi_attn_1_config.get('bias', True),
            q_noise=multi_attn_1_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_1_config.get('qn_block_size', 8),
            num_expert=multi_attn_1_config.get('num_expert', 4),
            head_dim=embed_dim_1 // num_heads_1,  # 自动计算head_dim
            use_attention_gate=multi_attn_1_config.get('use_attention_gate', True),
            cvloss=multi_attn_1_config.get('cvloss', 0.05),
            aux_loss=multi_attn_1_config.get('aux_loss', 1),
            zloss=multi_attn_1_config.get('zloss', 0.001),
            sample_topk=multi_attn_1_config.get('sample_topk', 0),
            noisy_gating=multi_attn_1_config.get('noisy_gating', True),
            use_pos_bias=multi_attn_1_config.get('use_pos_bias', False)
        )

        # LayerNorm for attention residual
        self.attn_norm = nn.LayerNorm(multi_attn_1_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_1_config = config.get('moe_1', {})

        # 创建MoE实例
        self.moe_1 = MoE(
            input_size=moe_1_config.get('input_size', 128),
            head_size=moe_1_config.get('head_size', 256),
            hidden_sizes=moe_1_config.get('hidden_sizes', None),  # 支持多层FFN
            num_experts=moe_1_config.get('num_experts', 3),
            k=moe_1_config.get('k', 1),
            need_merge=moe_1_config.get('need_merge', False),
            cvloss=moe_1_config.get('cvloss', 0.05),
            aux_loss=moe_1_config.get('aux_loss', 0.01),
            zloss=moe_1_config.get('zloss', 0.001),
            bias=moe_1_config.get('bias', True),
            activation=nn.GELU(),
            noisy_gating=moe_1_config.get('noisy_gating', False)
        )

        # 专门为moe_output_1定义的归一化层
        self.moe_1_norm = nn.LayerNorm(moe_1_config.get('input_size', 128))

        # 获取MultiheadAttention2配置
        multi_attn_2_config = config.get('multi_attention_2', {})

        # 实例化MultiheadAttention2
        embed_dim_2 = multi_attn_2_config.get('embed_dim', 128)
        num_heads_2 = multi_attn_2_config.get('num_heads', 1)
        self.multi_attention_2 = MultiheadAttention(
            embed_dim=embed_dim_2,
            num_heads=num_heads_2,
            dropout=multi_attn_2_config.get('dropout', 0.1),
            bias=multi_attn_2_config.get('bias', True),
            q_noise=multi_attn_2_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_2_config.get('qn_block_size', 8),
            num_expert=multi_attn_2_config.get('num_expert', 3),
            head_dim=embed_dim_2 // num_heads_2,  # 自动计算head_dim
            use_attention_gate=multi_attn_2_config.get('use_attention_gate', False),
            cvloss=multi_attn_2_config.get('cvloss', 0.05),
            aux_loss=multi_attn_2_config.get('aux_loss', 0.01),
            zloss=multi_attn_2_config.get('zloss', 0.001),
            sample_topk=multi_attn_2_config.get('sample_topk', 1),
            noisy_gating=multi_attn_2_config.get('noisy_gating', False),
            use_pos_bias=multi_attn_2_config.get('use_pos_bias', False)
        )

        # LayerNorm for attention2 residual
        self.attn2_norm = nn.LayerNorm(multi_attn_2_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_2_config = config.get('moe_2', {})

        # 创建MoE实例
        self.moe_2 = MoE(
            input_size=moe_2_config.get('input_size', 128),
            head_size=moe_2_config.get('head_size', 256),
            hidden_sizes=moe_2_config.get('hidden_sizes', None),  # 支持多层FFN
            num_experts=moe_2_config.get('num_experts', 3),
            k=moe_2_config.get('k', 1),
            need_merge=moe_2_config.get('need_merge', True),
            cvloss=moe_2_config.get('cvloss', 0.05),
            aux_loss=moe_2_config.get('aux_loss', 0.01),
            zloss=moe_2_config.get('zloss', 0.001),
            bias=moe_2_config.get('bias', True),
            activation=nn.GELU(),
            noisy_gating=moe_2_config.get('noisy_gating', False)
        )

        # 专门为moe_output_1定义的归一化层
        self.moe_2_norm = nn.LayerNorm(moe_2_config.get('input_size', 128))

        # 实例化PatchNeighborSearcher
        self.patch_neighbor_searcher = PatchNeighborSearcher()

        multi_attn_3_config = config.get('multi_attention_3', {})

        # 实例化MultiheadAttention3
        embed_dim_3 = multi_attn_3_config.get('embed_dim', 128)
        num_heads_3 = multi_attn_3_config.get('num_heads', 1)
        self.multi_attention_3 = MultiheadAttention(
            embed_dim=embed_dim_3,
            num_heads=num_heads_3,
            dropout=multi_attn_3_config.get('dropout', 0.1),
            bias=multi_attn_3_config.get('bias', True),
            q_noise=multi_attn_3_config.get('q_noise', 0.05),
            qn_block_size=multi_attn_3_config.get('qn_block_size', 8),
            num_expert=multi_attn_3_config.get('num_expert', 3),
            head_dim=embed_dim_3 // num_heads_3,  # 自动计算head_dim
            use_attention_gate=multi_attn_3_config.get('use_attention_gate', False),
            cvloss=multi_attn_3_config.get('cvloss', 0.05),
            aux_loss=multi_attn_3_config.get('aux_loss', 0.01),
            zloss=multi_attn_3_config.get('zloss', 0.001),
            sample_topk=multi_attn_3_config.get('sample_topk', 1),
            noisy_gating=multi_attn_3_config.get('noisy_gating', False),
            use_pos_bias=multi_attn_3_config.get('use_pos_bias', True)
        )

        # LayerNorm for attention2 residual
        self.attn3_norm = nn.LayerNorm(multi_attn_3_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_3_config = config.get('moe_3', {})

        # 创建MoE实例
        self.moe_3 = MoE(
            input_size=moe_3_config.get('input_size', 128),
            head_size=moe_3_config.get('head_size', 256),
            hidden_sizes=moe_3_config.get('hidden_sizes', None),  # 支持多层FFN
            num_experts=moe_3_config.get('num_experts', 3),
            k=moe_3_config.get('k', 1),
            need_merge=moe_3_config.get('need_merge', True),
            cvloss=moe_3_config.get('cvloss', 0.05),
            aux_loss=moe_3_config.get('aux_loss', 0.01),
            zloss=moe_3_config.get('zloss', 0.001),
            bias=moe_3_config.get('bias', True),
            activation=nn.GELU(),
            noisy_gating=moe_3_config.get('noisy_gating', False)
        )

        # 归一化层
        self.moe_3_norm = nn.LayerNorm(moe_3_config.get('input_size', 128))

        multi_attn_4_config = config.get('multi_attention_4', {})

        # 实例化MultiheadAttention4（自注意力）
        embed_dim_4 = multi_attn_4_config.get('embed_dim', 128)
        num_heads_4 = multi_attn_4_config.get('num_heads', 1)
        self.multi_attention_4 = MultiheadAttention(
            embed_dim=embed_dim_4,
            num_heads=num_heads_4,
            dropout=multi_attn_4_config.get('dropout', 0.1),
            bias=multi_attn_4_config.get('bias', True),
            q_noise=multi_attn_4_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_4_config.get('qn_block_size', 8),
            num_expert=multi_attn_4_config.get('num_expert', 3),
            head_dim=embed_dim_4 // num_heads_4,  # 自动计算head_dim
            use_attention_gate=multi_attn_4_config.get('use_attention_gate', False),
            cvloss=multi_attn_4_config.get('cvloss', 0.05),
            aux_loss=multi_attn_4_config.get('aux_loss', 0.01),
            zloss=multi_attn_4_config.get('zloss', 0.001),
            sample_topk=multi_attn_4_config.get('sample_topk', 1),
            noisy_gating=multi_attn_4_config.get('noisy_gating', False),
            use_pos_bias=multi_attn_4_config.get('use_pos_bias', False)
        )

        # LayerNorm for attention2 residual
        self.attn4_norm = nn.LayerNorm(multi_attn_4_config.get('embed_dim', 128))

        # 获取ISP解码器配置
        isp_1_config = config.get('isp_1', {})
        isp_2_config = config.get('isp_2', {})
        isp_4_config = config.get('isp_4', {})

        # 实例化三个ISP解码器（移除isp_3，去噪锐化使用固定参数）
        self.isp_1 = ISPDecoder(
            latent_dim=isp_1_config.get('latent_dim', 128),
            param_dim=isp_1_config.get('param_dim', 64),
            hidden_dims=isp_1_config.get('hidden_dims', [256, 128, 64, 32]),
            dropout=isp_1_config.get('dropout', 0.1),
            use_identity_init=True  # 保持接近0的初始化
        )

        self.isp_2 = ISPDecoder(
            latent_dim=isp_2_config.get('latent_dim', 128),
            param_dim=isp_2_config.get('param_dim', 576),
            hidden_dims=isp_2_config.get('hidden_dims', [256, 384, 512, 576]),
            dropout=isp_2_config.get('dropout', 0.1),
            use_identity_init=True  # 保持接近0的初始化
        )

        self.isp_4 = ISPDecoder(
            latent_dim=isp_4_config.get('latent_dim', 128),
            param_dim=isp_4_config.get('param_dim', 64),
            hidden_dims=isp_4_config.get('hidden_dims', [256, 128, 64, 32]),
            dropout=isp_4_config.get('dropout', 0.1),
            use_identity_init=False,  # 使用随机初始化
            bias_init=1.0  # 饱和度参数初始化为1.0，让网络默认预测更强的增强
        )

        vgg_model = vgg16(pretrained=True).features[:16]
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.perceptual_loss = PerceptualLoss(vgg_model)
        self.perceptual_loss.eval()

        # 在初始化过程中配置CUDA优化
        if torch.cuda.is_available():
            # 启用cuDNN自动调优
            torch.backends.cudnn.benchmark = True
            # 让算法行为确定
            torch.backends.cudnn.deterministic = False

        # 初始化小波域去噪模块（放在增强链最后，不需要训练）
        # 极致去噪配置：因为之前1.8效果都很小，现在使用最强参数
        self.wavelet_denoising = WaveletDenoising(
            wavelet='sym4',              # Daubechies小波（比sym4去噪更激进）
            level=2,                    # 3层分解（处理更多频率层次，比2层更强）
            mode='soft',                # 软阈值（平滑去噪）
            y_threshold_scale=1.8,      # Y通道极强去噪（1.8效果小→提高到4.0）
            c_threshold_scale=0.7       # 色度通道极强去噪（0.7效果小→提高到2.0）
        )

        self.color_transform = ColorTransform(
            residual_scale=0.5,    # 增大残差连接的缩放因子
            use_residual=True      # 使用残差连接
        )

        # ==================== 反棋盘格模块初始化 ====================
        # 1. 增强FFN（为三种参数各创建独立实例）
        self.isp_final_1_mlp = EnhancedFFN(dim=1, hidden_dim=1, dropout=0.1)  # gamma专用
        self.isp_final_2_mlp = EnhancedFFN(dim=9, hidden_dim=9, dropout=0.1)  # 颜色校正
        self.isp_final_4_mlp = EnhancedFFN(dim=1, hidden_dim=1, dropout=0.1)  # 饱和度专用

        # 2. 参数空间反棋盘格模块
        self.param_smoother = MultiScaleParamSmoother(in_channels=9, residual_weight=0.1)
        self.boundary_smoother = BoundaryAwareSmoothing(in_channels=9, residual_weight=0.1)

        # 3. 图像空间反棋盘格模块（用于处理像素级参数）
        # gamma和饱和度都是1通道
        self.anti_checkerboard_gamma = AntiCheckerboardModule(channels=1, residual_weight=0.1)
        self.anti_checkerboard_saturation = AntiCheckerboardModule(channels=1, residual_weight=0.1)
        # 颜色校正是9通道
        self.anti_checkerboard_color = AntiCheckerboardModule(channels=9, residual_weight=0.1)

        print("=" * 60)
        print("反棋盘格模块已初始化")
        print("=" * 60)

    def forward(self, x, target=None, return_isp_params=False):
        """
        定义模型的前向传播过程

        Args:
            x: 输入的低光照图像
            target: 目标正常光照图像。训练模式下必须提供，否则无法计算损失
            return_isp_params: 是否返回ISP参数用于可视化（仅在非训练模式有效）

        Returns:
            训练模式: (enhanced_image, loss, reconstruction_loss)
            测试模式 (return_isp_params=False): enhanced_image
            测试模式 (return_isp_params=True): (enhanced_image, isp_params_dict)
        """
        # 使用混合精度计算
        with amp.autocast(enabled=torch.cuda.is_available()):
            # 如果训练模式下没有提供target，则抛出错误
            if self.training and target is None:
                raise ValueError("训练模式下必须提供target图像")

            batch_size, channels, height, width = x.shape

            # 确保输入张量连续
            x = x.contiguous()

            # 特征提取
            features, h_windows, w_windows = self.feature_extractor(x)

            window_size = self.patch_size * self.win_size  # 计算窗口大小


            # EispGeneratorFFN处理
            e_isp = self.emb_gen(features)

            # 确保e_isp连续性，优化线性层计算
            e_isp_cont = e_isp.contiguous()
            e_isp_big = self.e_isp_expand(e_isp_cont)

            # QKIspGenerator处理
            q_isp, k_isp = self.qk_gen(e_isp)

            # KVFeatureGenerator处理
            k_features, v_features = self.kv_gen(features)

            # MultiheadAttention处理attn_output_1[batch_size, patches, seq_len, input_size]
            attn_output_1, _, atten_aux_loss_1 = self.multi_attention_1(
                query=q_isp,
                key=k_features,
                value=v_features,
                k_isp=k_isp,
                need_weights=False,
                before_softmax=False,
                need_head_weights=False
            )

            # 归一化 & 残差连接
            attn_output_1 = self.attn_norm(attn_output_1) + q_isp

            # MoE处理
            moe_output_1, moe_aux_loss_1 = self.moe_1(
                x=attn_output_1,
                sample_topk=0,
                multiply_by_gates=True
            )

            # 归一化处理moe_output_1
            moe_output_1 = self.moe_1_norm(moe_output_1)  # 归一化操作
            moe_output_1 = self.moe_1.concat(moe_output_1, e_isp_big)
            moe_output_1 = moe_output_1.squeeze(2)

            # MultiheadAttention2处理
            attn_output_2, _, atten_aux_loss_2 = self.multi_attention_2(
                query=moe_output_1,
                key=moe_output_1,
                value=moe_output_1,
                k_isp=None,
                need_weights=False,
                before_softmax=False,
                need_head_weights=False
            )

            # 归一化处理attn_output_2
            attn_output_2 = self.attn2_norm(attn_output_2) + moe_output_1

            # MOE处理
            moe_output_2, moe_aux_loss_2 = self.moe_2(
                x=attn_output_2,
                sample_topk=0,
                multiply_by_gates=True
            )

            # 归一化处理moe_output_2+残差连接
            moe_output_2 = self.moe_2_norm(moe_output_2) + attn_output_2

            # PatchNeighborSearcher 模块
            neighbor_features = self.patch_neighbor_searcher(moe_output_2, h_windows, w_windows)

            # MultiheadAttention3处理
            attn_output_3, _, atten_aux_loss_3 = self.multi_attention_3(
                query=moe_output_2,
                key=neighbor_features,
                value=neighbor_features,
                k_isp=None,
                need_weights=False,
                before_softmax=False,
                need_head_weights=False
            )

            # 归一化处理attn_output_3
            attn_output_3 = self.attn3_norm(attn_output_3) + moe_output_2

            # MoE处理
            moe_output_3, moe_aux_loss_3 = self.moe_3(
                x=attn_output_3,
                sample_topk=0,
                multiply_by_gates=True
            )

            # 归一化处理moe_output_2
            moe_output_3 = self.moe_3_norm(moe_output_3) + attn_output_3

            # 自注意力处理
            attn_output_4, _, atten_aux_loss_4 = self.multi_attention_4(
                query=moe_output_3,
                key=moe_output_3,  # 自注意力：key和query相同
                value=moe_output_3,  # 自注意力：value和query相同
                k_isp=None,
                need_weights=False,
                before_softmax=False,
                need_head_weights=False
            )

            # 残差连接和归一化
            attn_output_4 = self.attn4_norm(attn_output_4) + moe_output_3 + 1e-6

            # 1. 合并前两个维度
            bsz, patches, k, head_dim = attn_output_4.shape
            attn_output_4 = attn_output_4.reshape(-1, k, head_dim)  # [bsz*patches, tgt_len, head_dim]

            # 2. 将原始的第三个维度（tgt_len）放到第一个位置
            attn_output_4 = attn_output_4.transpose(0, 1)  # [tgt_len, bsz*patches, head_dim]

            # 从转置后的张量中获取三个专家嵌入（移除isp_expert_3）
            isp_expert_1 = attn_output_4[0]
            isp_expert_2 = attn_output_4[1]
            isp_expert_4 = attn_output_4[2]  # 跳过index 2，直接使用index 3作为饱和度专家

            # 分别输入到三个解码器中
            isp_final_1 = self.isp_1(isp_expert_1)  # [num_windows, param_dim_1 = 64] gamma参数
            isp_final_2 = self.isp_2(isp_expert_2)  # [num_windows, param_dim_2 = 576] 联合校正参数
            isp_final_4 = self.isp_4(isp_expert_4)  # [num_windows, param_dim_4 = 64] 饱和度参数

            # ==================== 方案B：两阶段参数生成（Coarse插值 + Fine残差）====================

            # 1. Gamma参数处理 [num_windows, 64] -> [B, H, W, 1]
            isp_final_1 = isp_final_1.reshape(batch_size, h_windows * w_windows, 64)

            # 1.1 Coarse阶段：提取每个window的平均值作为基础平滑参数
            isp_final_1_coarse = isp_final_1.mean(dim=-1, keepdim=True)  # [B, num_windows, 1]
            isp_final_1_coarse = isp_final_1_coarse.reshape(batch_size, h_windows, w_windows, 1)
            isp_final_1_coarse = isp_final_1_coarse.permute(0, 3, 1, 2)  # [B, 1, h_windows, w_windows]
            # 双线性插值到像素级（平滑上采样，消除window边界）
            isp_final_1_coarse = F.interpolate(
                isp_final_1_coarse,
                size=(h_windows * 8, w_windows * 8),
                mode='bilinear',
                align_corners=False
            )
            isp_final_1_coarse = isp_final_1_coarse.permute(0, 2, 3, 1)  # [B, H, W, 1]

            # 1.2 Fine阶段：逐像素残差细节
            isp_final_1_fine = isp_final_1.reshape(batch_size, h_windows, w_windows, 64, 1)
            isp_final_1_fine = isp_final_1_fine.reshape(batch_size, h_windows, w_windows, 8, 8, 1)
            isp_final_1_fine = isp_final_1_fine.permute(0, 1, 3, 2, 4, 5)
            isp_final_1_fine = isp_final_1_fine.reshape(batch_size, h_windows * 8, w_windows * 8, 1)

            # 1.3 两阶段加权组合（70% coarse平滑 + 30% fine细节）
            gamma_map = 0.9 * isp_final_1_coarse + 0.1 * isp_final_1_fine
            # EnhancedFFN处理
            gamma_map = self.isp_final_1_mlp(gamma_map)
            # 反棋盘格处理（强化）
            gamma_map = self.anti_checkerboard_gamma(gamma_map)
            gamma_map = self.anti_checkerboard_gamma(gamma_map)  # 二次平滑

            # 2. 颜色校正参数处理 [num_windows, 576] -> [B, H, W, 9]
            isp_final_2 = isp_final_2.reshape(batch_size, h_windows * w_windows, 576)

            # 2.1 Coarse阶段：每个window的9维参数平均值
            isp_final_2 = isp_final_2.reshape(batch_size, h_windows * w_windows, 64, 9)
            isp_final_2_coarse = isp_final_2.mean(dim=2)  # [B, num_windows, 9]
            isp_final_2_coarse = isp_final_2_coarse.reshape(batch_size, h_windows, w_windows, 9)
            isp_final_2_coarse = isp_final_2_coarse.permute(0, 3, 1, 2)  # [B, 9, h_windows, w_windows]
            # 双线性插值到像素级
            isp_final_2_coarse = F.interpolate(
                isp_final_2_coarse,
                size=(h_windows * 8, w_windows * 8),
                mode='bilinear',
                align_corners=False
            )
            isp_final_2_coarse = isp_final_2_coarse.permute(0, 2, 3, 1)  # [B, H, W, 9]

            # 2.2 Fine阶段：逐像素残差细节
            isp_final_2_fine = isp_final_2.reshape(batch_size, h_windows, w_windows, 64, 9)
            isp_final_2_fine = isp_final_2_fine.reshape(batch_size, h_windows, w_windows, 8, 8, 9)
            isp_final_2_fine = isp_final_2_fine.permute(0, 1, 3, 2, 4, 5)
            isp_final_2_fine = isp_final_2_fine.reshape(batch_size, h_windows * 8, w_windows * 8, 9)

            # 2.3 两阶段加权组合（80% coarse + 20% fine，颜色更强调平滑）
            isp_final_2 = 0.9 * isp_final_2_coarse + 0.1 * isp_final_2_fine
            # EnhancedFFN处理
            isp_final_2 = self.isp_final_2_mlp(isp_final_2)
            # 参数空间反棋盘格处理（强化）
            isp_final_2_t = isp_final_2.permute(0, 3, 1, 2)  # [B, 9, H, W]
            isp_final_2_t = self.param_smoother(isp_final_2_t)  # 多尺度平滑
            isp_final_2_t = self.param_smoother(isp_final_2_t)  # 二次多尺度平滑
            isp_final_2_t = self.boundary_smoother(isp_final_2_t)  # 边界感知平滑
            isp_final_2 = isp_final_2_t.permute(0, 2, 3, 1)  # 转回 [B, H, W, 9]
            # 图像空间反棋盘格（强化）
            isp_final_2 = self.anti_checkerboard_color(isp_final_2)
            isp_final_2 = self.anti_checkerboard_color(isp_final_2)  # 二次平滑

            # 3. 饱和度参数处理 [num_windows, 64] -> [B, H, W, 1]
            isp_final_4 = isp_final_4.reshape(batch_size, h_windows * w_windows, 64)

            # 3.1 Coarse阶段：提取每个window的平均值
            isp_final_4_coarse = isp_final_4.mean(dim=-1, keepdim=True)  # [B, num_windows, 1]
            isp_final_4_coarse = isp_final_4_coarse.reshape(batch_size, h_windows, w_windows, 1)
            isp_final_4_coarse = isp_final_4_coarse.permute(0, 3, 1, 2)  # [B, 1, h_windows, w_windows]
            # 双线性插值到像素级
            isp_final_4_coarse = F.interpolate(
                isp_final_4_coarse,
                size=(h_windows * 8, w_windows * 8),
                mode='bilinear',
                align_corners=False
            )
            isp_final_4_coarse = isp_final_4_coarse.permute(0, 2, 3, 1)  # [B, H, W, 1]

            # 3.2 Fine阶段：逐像素残差细节
            isp_final_4_fine = isp_final_4.reshape(batch_size, h_windows, w_windows, 64, 1)
            isp_final_4_fine = isp_final_4_fine.reshape(batch_size, h_windows, w_windows, 8, 8, 1)
            isp_final_4_fine = isp_final_4_fine.permute(0, 1, 3, 2, 4, 5)
            isp_final_4_fine = isp_final_4_fine.reshape(batch_size, h_windows * 8, w_windows * 8, 1)

            # 3.3 两阶段加权组合（70% coarse + 30% fine）
            saturation_map = 0.9 * isp_final_4_coarse + 0.1 * isp_final_4_fine
            # EnhancedFFN处理
            saturation_map = self.isp_final_4_mlp(saturation_map)
            # 反棋盘格处理（强化）
            saturation_map = self.anti_checkerboard_saturation(saturation_map)
            saturation_map = self.anti_checkerboard_saturation(saturation_map)  # 二次平滑

            # ==================== 应用ISP增强（核心增强+小波去噪）====================

            # 将原始图像从[batch_size, 3, H, W]转换为[batch_size, H, W, 3]
            enhanced_image = x.permute(0, 2, 3, 1)  # [batch_size, H, W, 3]

            # 0. 适度去噪（注释掉：结果太模糊）
            # enhanced_image_final = self.adaptive_denoising(images=enhanced_image)

            # 1. 颜色校正 (像素级变换矩阵)
            enhanced_image_final = self.color_transform(I_source=enhanced_image,
                                                               pred_transform_params=isp_final_2)

            # 2. 像素级Gamma增强
            enhanced_image_final = gamma_correct_pixelwise(L=enhanced_image_final, gamma_map=gamma_map)

            # 3. 像素级饱和度增强
            enhanced_image_final = contrast_enhancement_pixelwise(image=enhanced_image_final, alpha_map=saturation_map)

            # 4. 极致小波域去噪（去除增强过程放大的雪花噪点）
            # 极强配置：db1小波 + level=3 + y_threshold=4.0 + c_threshold=2.0
            # enhanced_image_final = self.wavelet_denoising(images=enhanced_image_final)

            # 转换回标准格式 [B, 3, H, W] 用于损失计算和返回
            enhanced_image_final = enhanced_image_final.permute(0, 3, 1, 2)

            # 训练模式下才计算损失并返回
            if self.training:
                # 计算总的辅助损失
                auxiliary_loss = (atten_aux_loss_1 + moe_aux_loss_1 + atten_aux_loss_2 + \
                                  moe_aux_loss_2 + atten_aux_loss_3 + moe_aux_loss_3 + atten_aux_loss_4) / 7

                # 使用目标图像计算重建损失 (使用Charbonnier损失提升鲁棒性)
                charbonnier_loss = CharbonnierLoss(epsilon=1e-3)
                reconstruction_loss = charbonnier_loss(enhanced_image_final, target)

                # 感知损失
                perceptual = self.perceptual_loss(enhanced_image_final, target)

                # LAB色彩损失 - 增加预防性检查
                def safe_lab_loss(pred, target):
                    """安全的LAB损失计算，避免CUDA索引越界"""
                    try:
                        # 预检查：确保输入有效
                        if pred.numel() == 0 or target.numel() == 0:
                            return torch.tensor(0.0, device=pred.device, requires_grad=True)

                        if pred.shape != target.shape:
                            return torch.tensor(0.0, device=pred.device, requires_grad=True)

                        # 确保数值在安全范围内
                        pred_safe = torch.clamp(pred, 0.0, 1.0)
                        target_safe = torch.clamp(target, 0.0, 1.0)

                        # 检查是否包含NaN或Inf
                        if torch.isnan(pred_safe).any() or torch.isinf(pred_safe).any():
                            return torch.tensor(0.0, device=pred.device, requires_grad=True)
                        if torch.isnan(target_safe).any() or torch.isinf(target_safe).any():
                            return torch.tensor(0.0, device=pred.device, requires_grad=True)

                        # 尝试LAB损失计算
                        lab_color_loss = LABColorLoss(color_weight=1.0)
                        return lab_color_loss(pred_safe, target_safe)

                    except Exception as e:
                        # 如果仍然失败，返回简单的L1损失
                        return F.l1_loss(pred, target, reduction='mean') * 0.01

                color_loss = safe_lab_loss(enhanced_image_final, target)

                # 计算PSNR损失 (添加数值稳定性检查)
                psnr_value = kornia.metrics.psnr(enhanced_image_final, target, max_val=1.0)
                # 限制PSNR值范围，避免极端情况
                psnr_value = torch.clamp(psnr_value, min=0.0, max=50.0)
                # 使用更稳定的损失计算方式
                psnr_loss = torch.max(torch.tensor(0.0, device=psnr_value.device),
                                      (50.0 - psnr_value) / 50.0)

                # 计算SSIM损失 (添加数值稳定性检查)
                ssim_value = kornia.metrics.ssim(enhanced_image_final, target, window_size=11, max_val=1.0)
                ssim_value = torch.clamp(ssim_value, min=0.0, max=1.0)  # 限制SSIM值范围
                ssim_loss = 1.0 - ssim_value.mean()  # SSIM越大损失越小

                # 检查损失是否为nan或inf
                if torch.isnan(psnr_loss) or torch.isinf(psnr_loss):
                    psnr_loss = torch.tensor(0.0, device=psnr_loss.device)
                if torch.isnan(ssim_loss) or torch.isinf(ssim_loss):
                    ssim_loss = torch.tensor(0.0, device=ssim_loss.device)

                # 计算每种损失的加权值
                weighted_reconstruction = self.reconstruction_weight * reconstruction_loss
                weighted_perceptual = self.perceptual_weight * perceptual
                weighted_auxiliary = self.auxiliary_weight * auxiliary_loss
                weighted_psnr = self.psnr_weight * psnr_loss
                weighted_ssim = self.ssim_weight * ssim_loss
                weighted_color = self.lab_color_weight * color_loss

                # 使用权重系数组合不同的损失 (L1+感知+辅助+PSNR+SSIM+LAB色彩)
                loss = (weighted_reconstruction + weighted_perceptual + weighted_auxiliary +
                        weighted_psnr + weighted_ssim + weighted_color)

                # 打印每种损失的加权值
                print(f"Loss Details - Reconstruction: {weighted_reconstruction.item():.6f}, "
                      f"Perceptual: {weighted_perceptual.item():.6f}, "
                      f"Auxiliary: {weighted_auxiliary.item():.6f}, "
                      f"PSNR: {weighted_psnr.item():.6f}, "
                      f"SSIM: {weighted_ssim.item():.6f}, "
                      f"Color: {weighted_color.item():.6f}, "
                      f"Total: {loss.item():.6f}")

                return enhanced_image_final, loss, reconstruction_loss
            else:
                # 测试模式
                if return_isp_params:
                    # 提取ISP参数用于热力图可视化（使用detach避免影响梯度）
                    with torch.no_grad():
                        # 1. Gamma参数：最终值 = 1.0 + Δ
                        final_gamma = 1.0 + gamma_map.squeeze(-1).detach()  # [B, H, W]

                        # 2. 颜色校正强度（尺度不变度量）
                        # isp_final_2: [B, H, W, 9] -> reshape to [B, H, W, 3, 3]
                        M = isp_final_2.reshape(batch_size, height, width, 3, 3).detach()

                        # 提取对角元素 [B, H, W, 3]
                        diag = torch.stack([M[:, :, :, 0, 0], M[:, :, :, 1, 1], M[:, :, :, 2, 2]], dim=-1)

                        # 提取非对角元素 [B, H, W, 6]
                        off_diag_elements = []
                        for i in range(3):
                            for j in range(3):
                                if i != j:
                                    off_diag_elements.append(M[:, :, :, i, j])
                        off_diag = torch.stack(off_diag_elements, dim=-1)

                        # 计算范数
                        diag_norm = torch.sqrt(torch.sum(diag**2, dim=-1))  # [B, H, W]
                        off_diag_norm = torch.sqrt(torch.sum(off_diag**2, dim=-1))  # [B, H, W]

                        # 计算强度比（尺度不变，色度/亮度比）
                        color_correction_strength = off_diag_norm / (diag_norm + 1e-8)  # [B, H, W]

                        # 3. 饱和度参数：最终值 = 1.5 + tanh(Δ) * 0.4
                        final_saturation = 1.5 + torch.tanh(saturation_map.squeeze(-1).detach()) * 0.4  # [B, H, W]

                        # 返回ISP参数字典
                        isp_params = {
                            'gamma_map': final_gamma,  # [B, H, W]
                            'color_correction_strength': color_correction_strength,  # [B, H, W]
                            'saturation_map': final_saturation  # [B, H, W]
                        }

                        return enhanced_image_final, isp_params
                else:
                    # 仅返回增强后的图像
                    return enhanced_image_final
