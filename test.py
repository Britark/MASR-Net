import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json
import warnings

# 抑制torch.cuda.amp.autocast的FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')

# 抑制torchvision模型权重加载的UserWarning（来自LPIPS库）
warnings.filterwarnings('ignore', category=UserWarning, message='.*Arguments other than a weight enum.*')
warnings.filterwarnings('ignore', category=UserWarning, message=".*The parameter 'pretrained' is deprecated.*")

# LPIPS导入
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告：lpips未安装，将跳过LPIPS指标计算。安装方法: pip install lpips")

# 导入自定义模块
from data_loader import get_data_loaders
from models import Model


def calculate_single_image_metrics(img1, img2, lpips_fn=None):
    """
    计算两张图像之间的PSNR、SSIM和LPIPS

    Args:
        img1: 第一张图像 numpy array (H, W, C)
        img2: 第二张图像 numpy array (H, W, C)
        lpips_fn: LPIPS计算函数（可选）

    Returns:
        psnr_value: PSNR值
        ssim_value: SSIM值
        lpips_value: LPIPS值（如果lpips_fn为None则返回-1）
    """
    # 确保数值在[0, 1]范围内
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    # 计算PSNR
    psnr_val = psnr(img1, img2, data_range=1.0)

    # 计算SSIM
    if img1.shape[-1] == 3:  # RGB图像
        ssim_val = ssim(img1, img2, data_range=1.0, channel_axis=-1)
    else:  # 灰度图像
        ssim_val = ssim(img1.squeeze(), img2.squeeze(), data_range=1.0)

    # 计算LPIPS
    lpips_val = -1.0
    if lpips_fn is not None:
        # 转换为tensor并调整到[-1, 1]范围，格式为[1, C, H, W]
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()

        # 从[0,1]转到[-1,1]
        img1_tensor = img1_tensor * 2.0 - 1.0
        img2_tensor = img2_tensor * 2.0 - 1.0

        # 移到GPU（如果LPIPS模型在GPU上）
        if next(lpips_fn.parameters()).is_cuda:
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()

        with torch.no_grad():
            lpips_val = lpips_fn(img1_tensor, img2_tensor).item()

    return psnr_val, ssim_val, lpips_val


def test_model(args):
    """
    在测试集上测试模型

    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 混合精度推理设置
    use_amp = args.use_amp and torch.cuda.is_available()
    if use_amp:
        print("启用混合精度推理 (AMP) - 速度更快")

    # 动态生成输出目录路径，根据权重文件关键字确定数据集类型
    weight_filename = os.path.basename(args.weights_path)  # 获取权重文件名
    weight_name = os.path.splitext(weight_filename)[0]     # 去掉扩展名

    # 根据权重文件名关键字确定数据集类型
    if "LOLv1" in weight_filename or "LOL_V1" in weight_filename:
        dataset_type = "LOLv1"
    elif "LOLv2" in weight_filename or "LOL_v2" in weight_filename:
        dataset_type = "LOLv2"
    elif "LSRW" in weight_filename:
        dataset_type = "LSRW"
    else:
        # 如果无法识别，使用原始方法
        dataset_type = weight_name

    args.output_dir = f"./result/{dataset_type}"  # 构造新的输出目录名

    # 创建保存结果的目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取数据加载器
    root_dir = os.path.expanduser(args.data_dir)
    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 根据参数选择使用哪个数据集
    if args.dataset_split == 'train':
        data_loader = train_loader
        print("Testing on training set")
    elif args.dataset_split == 'val':
        data_loader = val_loader
        print("Testing on validation set")
    else:
        data_loader = test_loader
        print("Testing on test set")

    # 初始化模型
    model = Model().to(device)

    # 加载模型权重
    print(f"Loading model weights from {args.weights_path}")
    if args.weights_path.endswith('_weights.pth'):
        # 加载只包含权重的文件
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    else:
        # 加载包含完整训练状态的检查点
        checkpoint = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 设置为评估模式
    model.eval()

    # 初始化LPIPS模型（根据参数决定是否使用）
    lpips_fn = None
    if LPIPS_AVAILABLE and not args.skip_lpips:
        print("初始化LPIPS模型 (使用AlexNet)...")
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_fn.eval()
        print("LPIPS模型已加载")
        if use_amp:
            print("  提示：LPIPS计算较慢，建议使用 --skip-lpips 跳过以提升速度")
    else:
        if args.skip_lpips:
            print("跳过LPIPS计算（用户指定 --skip-lpips）")
        else:
            print("跳过LPIPS计算（未安装lpips库）")

    # 进度条
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing model")

    # 图像计数
    image_count = 0

    # 用于记录总体指标
    total_enhanced_psnr = 0
    total_enhanced_ssim = 0
    total_enhanced_lpips = 0
    total_original_psnr = 0
    total_original_ssim = 0
    total_original_lpips = 0
    total_images = 0

    with torch.no_grad():
        for batch_idx, batch_data in progress_bar:
            # 获取低光照输入
            inputs = batch_data['input'].to(device, non_blocking=True)

            # 可以用于比较的参考图像（groundtruth）
            targets = batch_data['target'].to(device, non_blocking=True) if not args.save_only else batch_data['target']

            # 推理 - 使用混合精度加速
            if use_amp:
                with torch.cuda.amp.autocast():
                    enhanced_images = model(inputs)
            else:
                enhanced_images = model(inputs)

            # 如果只保存图像不计算指标，快速处理
            if args.save_only:
                for i in range(inputs.shape[0]):
                    enhanced_img = enhanced_images[i].cpu().permute(1, 2, 0).numpy()
                    enhanced_img = np.clip(enhanced_img, 0, 1)
                    enhanced_img_pil = Image.fromarray((enhanced_img * 255).astype(np.uint8))
                    enhanced_path = os.path.join(args.output_dir, f"enhanced_{image_count:04d}.png")
                    enhanced_img_pil.save(enhanced_path)
                    image_count += 1
                progress_bar.set_postfix({'saved': image_count})
                continue

            # 批量计算指标（在GPU上，更快）
            from skimage.metrics import peak_signal_noise_ratio as psnr_cpu
            from skimage.metrics import structural_similarity as ssim_cpu

            # 逐个处理图像以计算指标
            for i in range(inputs.shape[0]):
                # 只在需要时转换到CPU
                enhanced_img = enhanced_images[i].cpu().permute(1, 2, 0).numpy()
                enhanced_img = np.clip(enhanced_img, 0, 1)

                normal_light_img = targets[i].cpu().permute(1, 2, 0).numpy()
                normal_light_img = np.clip(normal_light_img, 0, 1)

                # 计算增强图像指标
                enhanced_psnr = psnr_cpu(enhanced_img, normal_light_img, data_range=1.0)
                if enhanced_img.shape[-1] == 3:
                    enhanced_ssim = ssim_cpu(enhanced_img, normal_light_img, data_range=1.0, channel_axis=-1)
                else:
                    enhanced_ssim = ssim_cpu(enhanced_img.squeeze(), normal_light_img.squeeze(), data_range=1.0)

                # LPIPS计算（如果启用）
                enhanced_lpips = -1.0
                if lpips_fn is not None:
                    img_tensor = torch.from_numpy(enhanced_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
                    target_tensor = torch.from_numpy(normal_light_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
                    img_tensor = img_tensor * 2.0 - 1.0
                    target_tensor = target_tensor * 2.0 - 1.0
                    with torch.no_grad():
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                enhanced_lpips = lpips_fn(img_tensor, target_tensor).item()
                        else:
                            enhanced_lpips = lpips_fn(img_tensor, target_tensor).item()

                # 只在需要时计算原始图像指标
                if not args.skip_original_metrics:
                    low_light_img = inputs[i].cpu().permute(1, 2, 0).numpy()
                    low_light_img = np.clip(low_light_img, 0, 1)
                    original_psnr = psnr_cpu(low_light_img, normal_light_img, data_range=1.0)
                    if low_light_img.shape[-1] == 3:
                        original_ssim = ssim_cpu(low_light_img, normal_light_img, data_range=1.0, channel_axis=-1)
                    else:
                        original_ssim = ssim_cpu(low_light_img.squeeze(), normal_light_img.squeeze(), data_range=1.0)

                    original_lpips = -1.0
                    if lpips_fn is not None:
                        low_tensor = torch.from_numpy(low_light_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
                        low_tensor = low_tensor * 2.0 - 1.0
                        with torch.no_grad():
                            if use_amp:
                                with torch.cuda.amp.autocast():
                                    original_lpips = lpips_fn(low_tensor, target_tensor).item()
                            else:
                                original_lpips = lpips_fn(low_tensor, target_tensor).item()

                    total_original_psnr += original_psnr
                    total_original_ssim += original_ssim
                    if original_lpips >= 0:
                        total_original_lpips += original_lpips
                else:
                    original_psnr = 0
                    original_ssim = 0
                    original_lpips = -1

                # 累积指标
                total_enhanced_psnr += enhanced_psnr
                total_enhanced_ssim += enhanced_ssim
                if enhanced_lpips >= 0:
                    total_enhanced_lpips += enhanced_lpips
                total_images += 1

                # 保存增强图像（如果启用）
                if args.save_enhanced:
                    enhanced_img_pil = Image.fromarray((enhanced_img * 255).astype(np.uint8))
                    enhanced_path = os.path.join(args.output_dir, f"enhanced_{image_count:04d}.png")
                    enhanced_img_pil.save(enhanced_path)

                image_count += 1

                # 更新进度条
                postfix_dict = {
                    'enh_psnr': f"{enhanced_psnr:.2f}",
                    'enh_ssim': f"{enhanced_ssim:.4f}",
                }
                if not args.skip_original_metrics:
                    postfix_dict['orig_psnr'] = f"{original_psnr:.2f}"
                    postfix_dict['orig_ssim'] = f"{original_ssim:.4f}"
                if enhanced_lpips >= 0:
                    postfix_dict['enh_lpips'] = f"{enhanced_lpips:.4f}"
                    if not args.skip_original_metrics and original_lpips >= 0:
                        postfix_dict['orig_lpips'] = f"{original_lpips:.4f}"
                progress_bar.set_postfix(postfix_dict)

    # 计算平均指标
    avg_enhanced_psnr = total_enhanced_psnr / total_images if total_images > 0 else 0
    avg_enhanced_ssim = total_enhanced_ssim / total_images if total_images > 0 else 0
    avg_original_psnr = total_original_psnr / total_images if total_images > 0 and not args.skip_original_metrics else 0
    avg_original_ssim = total_original_ssim / total_images if total_images > 0 and not args.skip_original_metrics else 0

    # LPIPS平均值（如果可用）
    avg_enhanced_lpips = -1.0
    avg_original_lpips = -1.0
    if LPIPS_AVAILABLE and total_enhanced_lpips > 0:
        avg_enhanced_lpips = total_enhanced_lpips / total_images
        avg_original_lpips = total_original_lpips / total_images

    # 保存测试结果
    save_test_results(avg_enhanced_psnr, avg_enhanced_ssim, avg_enhanced_lpips,
                     avg_original_psnr, avg_original_ssim, avg_original_lpips,
                     total_images, args.output_dir)

    print(f"\n=== 测试结果 ===")
    print(f"处理图像总数: {total_images}")
    if not args.skip_original_metrics:
        print(f"原始低光照图像平均 PSNR: {avg_original_psnr:.2f} dB")
        print(f"原始低光照图像平均 SSIM: {avg_original_ssim:.4f}")
        if avg_original_lpips >= 0:
            print(f"原始低光照图像平均 LPIPS: {avg_original_lpips:.4f}")
    print(f"增强后图像平均 PSNR: {avg_enhanced_psnr:.2f} dB")
    print(f"增强后图像平均 SSIM: {avg_enhanced_ssim:.4f}")
    if avg_enhanced_lpips >= 0:
        print(f"增强后图像平均 LPIPS: {avg_enhanced_lpips:.4f}")
    if not args.skip_original_metrics:
        print(f"PSNR 提升: {avg_enhanced_psnr - avg_original_psnr:.2f} dB")
        print(f"SSIM 提升: {avg_enhanced_ssim - avg_original_ssim:.4f}")
        if avg_enhanced_lpips >= 0:
            print(f"LPIPS 改善: {avg_original_lpips - avg_enhanced_lpips:.4f} (↓更好)")

    print(f"\nTesting completed. Results saved in {args.output_dir}")


def save_test_results(avg_enhanced_psnr, avg_enhanced_ssim, avg_enhanced_lpips,
                     avg_original_psnr, avg_original_ssim, avg_original_lpips,
                     total_images, output_dir):
    """保存测试结果到文件"""
    results_file = os.path.join(output_dir, "test_results.txt")
    json_file = os.path.join(output_dir, "test_results.json")

    # 保存文本格式
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== 图像增强测试结果 ===\n\n")
        f.write(f"处理图像总数: {total_images}\n\n")
        f.write("--- 原始低光照图像 ---\n")
        f.write(f"平均 PSNR: {avg_original_psnr:.2f} dB\n")
        f.write(f"平均 SSIM: {avg_original_ssim:.4f}\n")
        if avg_original_lpips >= 0:
            f.write(f"平均 LPIPS: {avg_original_lpips:.4f}\n")
        f.write("\n--- 增强后图像 ---\n")
        f.write(f"平均 PSNR: {avg_enhanced_psnr:.2f} dB\n")
        f.write(f"平均 SSIM: {avg_enhanced_ssim:.4f}\n")
        if avg_enhanced_lpips >= 0:
            f.write(f"平均 LPIPS: {avg_enhanced_lpips:.4f}\n")
        f.write("\n--- 提升/改善 ---\n")
        f.write(f"PSNR 提升: {avg_enhanced_psnr - avg_original_psnr:.2f} dB\n")
        f.write(f"SSIM 提升: {avg_enhanced_ssim - avg_original_ssim:.4f}\n")
        if avg_enhanced_lpips >= 0:
            f.write(f"LPIPS 改善: {avg_original_lpips - avg_enhanced_lpips:.4f} (↓更好)\n")

    # 保存JSON格式
    results_data = {
        'total_images': total_images,
        'avg_original_psnr': float(avg_original_psnr),
        'avg_original_ssim': float(avg_original_ssim),
        'avg_enhanced_psnr': float(avg_enhanced_psnr),
        'avg_enhanced_ssim': float(avg_enhanced_ssim),
        'psnr_improvement': float(avg_enhanced_psnr - avg_original_psnr),
        'ssim_improvement': float(avg_enhanced_ssim - avg_original_ssim)
    }

    if avg_enhanced_lpips >= 0:
        results_data['avg_original_lpips'] = float(avg_original_lpips)
        results_data['avg_enhanced_lpips'] = float(avg_enhanced_lpips)
        results_data['lpips_improvement'] = float(avg_original_lpips - avg_enhanced_lpips)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"详细结果已保存到: {results_file}")
    print(f"JSON格式结果已保存到: {json_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test low-light image enhancement model")

    # 基本参数
    # 数据集路径选择:
    # LOLv1: ./datasets/LOL_V1/lol_dataset
    # LOLv2: ./datasets/LOL-v2
    # LSRW:  ./datasets/OpenDataLab___LSRW/raw/LSRW
    parser.add_argument('--data_dir', type=str,
                        default="../swin-MOA/LOL-v2-Dataset/archive/LOL-v2/Real_captured",
                        help='Dataset path')

    # 模型权重路径选择:
    # LOLv1: ./checkpoints/LOLv1_checkpoints.pth
    # LOLv2: ./checkpoints/LOLv2_real_checkpoints.pth
    # LSRW:  ./checkpoints/LSRW_checkpoints.pth
    parser.add_argument('--weights_path', type=str,
                        default="./checkpoints/best_model.pth",
                        help='Model weights path')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Test batch size (增大可提升速度)')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='Number of worker threads for data loading')
    parser.add_argument('--dataset_split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which dataset split to test on')
    parser.add_argument('--output_dir', type=str, default="./test_results",
                        help='Directory to save enhanced images')

    # 保存和计算选项
    parser.add_argument('--save_enhanced', action='store_true',
                        default=True,
                        help='Whether to save enhanced images separately')
    parser.add_argument('--save_only', action='store_true',
                        help='只保存图像，不计算指标（最快模式）')

    # 性能优化选项
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度推理（速度更快，推荐）')
    parser.add_argument('--skip_lpips', action='store_true',
                        help='跳过LPIPS计算（大幅提升速度）')
    parser.add_argument('--skip_original_metrics', action='store_true',
                        help='跳过原始图像指标计算（提升速度）')

    args = parser.parse_args([])  # 传入空列表避免从命令行读取参数

    print(f"=== 测试配置 ===")
    print(f"数据集: {args.dataset_split}")
    print(f"模型权重: {args.weights_path}")
    print(f"输出目录: {args.output_dir}")

    test_model(args)
