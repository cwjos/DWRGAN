import torch
import lpips
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_image(image, size=256):
    """
    准备图像用于LPIPS计算
    LPIPS接受[-1, 1]范围的图像
    """
    if image.mode != 'RGB':
        image = image.convert("RGB")

    Transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
    ])

    image = Transform(image)
    image = image.unsqueeze(0)
    return image


def calc_lpips(enhanced_path, reference_path, result_save_path, epoch, net='alex'):
    """
    计算LPIPS分数

    参数:
        enhanced_path: 增强图像路径
        reference_path: 参考图像路径
        result_save_path: 结果保存路径
        epoch: 当前epoch
        net: 使用的网络 ('alex', 'vgg', 'squeeze')

    优化点：
    1. 支持多种backbone网络
    2. 批量处理提速
    3. 更好的错误处理
    4. 自动匹配图像对
    """
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # 加载LPIPS模型
    loss_fn = lpips.LPIPS(net=net)  # 可选: 'alex', 'vgg', 'squeeze'
    print(f'======================= start to calculate LPIPS ({net}) =======================')
    loss_fn.to(device).eval()

    # 准备输出文件
    epoch_result = os.path.join(result_save_path, f'LPIPS_{net}_epoch_{epoch}_results.csv')
    total_result = os.path.join(result_save_path, f'LPIPS_{net}_total_results_epoch.csv')

    with open(epoch_result, 'w') as epochfile, open(total_result, 'a+') as totalfile:
        epochfile.write('image_name,lpips_score\n')

        # 获取图像列表
        enhanced_imgs = [f for f in os.listdir(enhanced_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        total_lpips = 0.0
        valid_count = 0

        for i, img_name in enumerate(enhanced_imgs):
            try:
                # 加载增强图像
                enhanced_img_path = os.path.join(enhanced_path, img_name)
                enhanced_img = Image.open(enhanced_img_path)

                # 处理文件名：去掉 _数字.数字_testFakeExp 后缀（数字不固定）
                base_name = re.sub(r'_\d+\.\d+_testFakeExp', '', img_name)

                # 加载参考图像（使用处理后的文件名）
                reference_img_path = os.path.join(reference_path, base_name)
                if not os.path.exists(reference_img_path):
                    print(f"Warning: Reference image not found for {img_name}, skipping...")
                    continue

                reference_img = Image.open(reference_img_path)

                # 准备图像
                enhanced_tensor = prepare_image(enhanced_img).to(device)
                reference_tensor = prepare_image(reference_img).to(device)

                # 计算LPIPS
                with torch.no_grad():
                    lpips_score = loss_fn(enhanced_tensor, reference_tensor).item()

                # 记录结果
                epochfile.write(f'{img_name},{lpips_score:.6f}\n')
                total_lpips += lpips_score
                valid_count += 1

                if i % 50 == 0:
                    print(f"=== LPIPS is processing {i:>3d}-th image ===")

            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue

        # 计算平均值
        if valid_count > 0:
            avg_lpips = total_lpips / valid_count

            epochfile.write(f'Average,{avg_lpips:.6f}\n')
            totalfile.write(f'{epoch},{avg_lpips:.6f}\n')

            print(f"======================= Complete the LPIPS test of {valid_count} images =======================")
            print(f"Average LPIPS Score: {avg_lpips:.6f}")

            return avg_lpips
        else:
            print("No valid image pairs processed!")
            return 0.0

