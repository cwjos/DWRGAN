# -*- coding: utf-8 -*-
import os
import math
import numpy as np
# import skimage.test.peak_signal_noise_ratio as psnr_skimage
import cv2
import glob
import datetime
from PIL import Image


# def center_crop(img, target_h, target_w):
#     h, w = img.shape[:2]
#     if h < target_h or w < target_w:
#         raise ValueError("Cannot center-crop: target larger than image.")
#     start_y = (h - target_h) // 2
#     start_x = (w - target_w) // 2
#     return img[start_y:start_y + target_h, start_x:start_x + target_w]
#
#
# def resize_lanczos_float01(img_float01, out_w, out_h):
#     is_gray = (img_float01.ndim == 2)
#     if img_float01.max() <= 1.0:
#         img_u8 = (np.clip(img_float01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
#     else:
#         img_u8 = np.clip(img_float01.round(), 0, 255).astype(np.uint8)
#
#     if is_gray:
#         pil = Image.fromarray(img_u8, mode='L')
#     else:
#         pil = Image.fromarray(img_u8)
#
#     pil2 = pil.resize((out_w, out_h), resample=Image.LANCZOS)
#     arr = np.asarray(pil2).astype(np.float32) / 255.0
#     return arr
#
#
# def bgr2ycbcr(img, only_y=False):
#     # 输入：RGB float [0,1] 或 uint8 0..255
#     if img.dtype == np.uint8:
#         img_f = img.astype(np.float32) / 255.0
#     else:
#         img_f = img.astype(np.float32)
#         if img_f.max() > 1.5:
#             img_f = img_f / 255.0
#
#     R = img_f[..., 0]
#     G = img_f[..., 1]
#     B = img_f[..., 2]
#     Y = 0.299 * R + 0.587 * G + 0.114 * B
#
#     if only_y:
#         return Y
#     else:
#         Cb = 0.5 * (B - Y) / (1 - 0.114) + 0.5
#         Cr = 0.5 * (R - Y) / (1 - 0.299) + 0.5
#         return np.stack([Y, Cb, Cr], axis=-1)
#
#
# def calculate_psnr(img1, img2, data_range=255.0):
#     def to_float64_scaled(x):
#         if x.dtype == np.uint8:
#             return x.astype(np.float64)
#         x = x.astype(np.float64)
#         if x.max() <= 1.1:
#             return x * data_range
#         else:
#             return x
#     a = to_float64_scaled(img1)
#     b = to_float64_scaled(img2)
#     if a.shape != b.shape:
#         raise ValueError(f"calculate_psnr: shape mismatch {a.shape} vs {b.shape}")
#     mse = np.mean((a - b) ** 2, dtype=np.float64)
#     if mse == 0:
#         return float('inf')
#     return 10.0 * math.log10((data_range ** 2) / mse)
#
#
# # ---------------- ECC 配准相关 ----------------
# def ecc_align(src_rgb_f01, ref_rgb_f01, warp_mode='affine', number_of_iterations=500, termination_eps=1e-6):
#     """
#     将 src_rgb_f01 对齐到 ref_rgb_f01，输入均为 RGB float in [0,1]。
#     返回：aligned_src_rgb_f01（与 ref 尺寸一致）。
#     warp_mode: 'affine' 或 'homography'
#     注意：函数内部做了异常捕获，若失败会抛异常（调用方可捕获并回退）。
#     """
#     # 转 uint8 再到灰度 float32（findTransformECC 要求 single-channel float32）
#     def to_gray_float32(img_f01):
#         if img_f01.dtype != np.float32:
#             img_f01 = img_f01.astype(np.float32)
#         gray = cv2.cvtColor((np.clip(img_f01,0,1)*255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#         return gray.astype(np.float32) / 255.0
#
#     src_h, src_w = src_rgb_f01.shape[:2]
#     ref_h, ref_w = ref_rgb_f01.shape[:2]
#     # 若尺寸不同，请先 resize src 到 ref（使用 LANCZOS），调用方在本实现中已执行
#     src_gray = to_gray_float32(src_rgb_f01)
#     ref_gray = to_gray_float32(ref_rgb_f01)
#
#     if warp_mode == 'homography':
#         warp_matrix = np.eye(3, 3, dtype=np.float32)
#         cv_warp_mode = cv2.MOTION_HOMOGRAPHY
#     else:
#         warp_matrix = np.eye(2, 3, dtype=np.float32)
#         cv_warp_mode = cv2.MOTION_AFFINE
#
#     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
#
#     # findTransformECC(templateImage, inputImage, warp_matrix,...)
#     # 注意顺序：template = ref_gray, input = src_gray
#     (cc, warp_matrix) = cv2.findTransformECC(ref_gray, src_gray, warp_matrix, cv_warp_mode, criteria, inputMask=None, gaussFiltSize=1)
#
#     # warp back src to ref size
#     if cv_warp_mode == cv2.MOTION_HOMOGRAPHY:
#         aligned = cv2.warpPerspective((np.clip(src_rgb_f01,0,1)*255.0).astype(np.uint8), warp_matrix, (ref_w, ref_h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#     else:
#         aligned = cv2.warpAffine((np.clip(src_rgb_f01,0,1)*255.0).astype(np.uint8), warp_matrix, (ref_w, ref_h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#
#     aligned_f01 = aligned.astype(np.float32) / 255.0
#     return aligned_f01
#
#
# # #--------------- 主函数：带 ECC 的 calc_psnr ----------------
# def calc_psnr(folder_Gen, folder_GT, result_save_path, epoch,
#               on_mismatch='crop', use_ecc=True, ecc_warp='affine'):
#     """
#     增强版 calc_psnr，新增参数：
#       - use_ecc: 是否尝试在对齐前用 ECC 把 GT 对齐到 Gen（默认 False）
#       - ecc_warp: 'affine' 或 'homography'（默认 'affine'）
#     其它行为同之前（on_mismatch: 'raise'/'resize'/'crop'）。
#     """
#     if not os.path.exists(result_save_path):
#         os.makedirs(result_save_path)
#
#     total_psnr = 0.0
#     processed = 0
#     epoch_result = os.path.join(result_save_path, f'PSNR_epoch_{epoch}.csv')
#     total_result = os.path.join(result_save_path, 'PSNR_total_results_epoch_avgpsnr.csv')
#
#     epochfile = open(epoch_result, 'w')
#     epochfile.write('image_name,psnr\n')
#     totalfile = open(total_result, 'a+')
#
#     crop_border = 0
#     test_Y = True
#     img_list = sorted(glob.glob(os.path.join(folder_Gen, '*')))
#
#     if test_Y:
#         print('Testing Y channel (luminance only).')
#     else:
#         print('Testing RGB channels.')
#
#     starttime = datetime.datetime.now()
#     mismatch_reported = False
#
#     for idx, img_path in enumerate(img_list):
#         gen_bgr = cv2.imread(img_path)
#         if gen_bgr is None:
#             print(f"[Warning] Failed to read generated image: {img_path}")
#             continue
#         im_Gen = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
#
#         base_name = os.path.splitext(os.path.basename(img_path))[0]
#         # print(base_name)
#         try:
#             imgName, _, _ = base_name.rsplit('_', 2)
#         except ValueError:
#             imgName = base_name
#         GT_imgName = imgName + '.png'
#         GT_path = os.path.join(folder_GT, GT_imgName)
#
#         if not os.path.exists(GT_path):
#             print(f"[Warning] GT image not found for {GT_imgName}")
#             continue
#
#         gt_bgr = cv2.imread(GT_path)
#         if gt_bgr is None:
#             print(f"[Warning] Failed to read GT image: {GT_path}")
#             continue
#         im_GT = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
#
#         # 如果启用 ECC：先把 GT resize 到 Gen 尺寸（LANCZOS），再尝试配准（GT 对齐到 Gen）
#         if use_ecc:
#             try:
#                 # 先把 GT resize 到 Gen 的尺寸（降低求解难度）
#                 if im_GT.shape[:2] != im_Gen.shape[:2]:
#                     im_GT_resz = resize_lanczos_float01(im_GT, im_Gen.shape[1], im_Gen.shape[0])
#                 else:
#                     im_GT_resz = im_GT
#                 # 尝试 ECC 对齐（将 GT 对齐到 Gen）
#                 im_GT_aligned = ecc_align(im_GT_resz, im_Gen, warp_mode=ecc_warp)
#                 # 如果配准成功，使用配准后的 GT 继续后续流程（此时 GT 与 Gen 尺寸一致）
#                 im_GT = im_GT_aligned
#                 if not mismatch_reported:
#                     print(f"[Info] ECC alignment succeeded for {GT_imgName} (warp={ecc_warp}).")
#                     mismatch_reported = True
#             except Exception as e:
#                 # 配准失败：打印日志但不终止，继续后续常规对齐策略
#                 print(f"[Warning] ECC alignment failed for {GT_imgName}: {e}. Falling back to normal handling.")
#                 # 继续，不做任何替换（im_GT 保持原样）
#
#         # 如果不相同，继续按 on_mismatch 策略处理
#         if im_Gen.shape != im_GT.shape:
#             h_g, w_g = im_Gen.shape[:2]
#             h_t, w_t = im_GT.shape[:2]
#
#             if on_mismatch == 'raise':
#                 raise ValueError(f"Image shape mismatch: {im_Gen.shape} vs {im_GT.shape}")
#             elif on_mismatch == 'resize':
#                 # 缩放 Gen 到 GT（LANCZOS）
#                 im_Gen = resize_lanczos_float01(im_Gen, w_t, h_t)
#                 if not mismatch_reported:
#                     print(f"[Info] Resized Gen ({w_g}x{h_g}) -> GT ({w_t}x{h_t}) for {GT_imgName} using LANCZOS")
#                     mismatch_reported = True
#             elif on_mismatch == 'crop':
#                 min_h = min(h_g, h_t)
#                 min_w = min(w_g, w_t)
#                 try:
#                     if (h_g, w_g) != (min_h, min_w):
#                         im_Gen = center_crop(im_Gen, min_h, min_w)
#                     if (h_t, w_t) != (min_h, min_w):
#                         im_GT = center_crop(im_GT, min_h, min_w)
#                     if not mismatch_reported:
#                         print(f"[Info] Center-cropped to ({min_h}x{min_w}) for {GT_imgName}")
#                         mismatch_reported = True
#                 except Exception:
#                     # 若裁剪失败，则退回到 LANCZOS resize
#                     im_Gen = resize_lanczos_float01(im_Gen, w_t, h_t)
#                     if not mismatch_reported:
#                         print(f"[Info] Center-crop failed; resized Gen ({w_g}x{h_g}) -> GT ({w_t}x{h_t}) for {GT_imgName} using LANCZOS")
#                         mismatch_reported = True
#             else:
#                 raise ValueError("Invalid on_mismatch option. Choose 'raise', 'resize' or 'crop'.")
#
#         # Y 通道
#         if test_Y:
#             im_GT_proc = bgr2ycbcr(im_GT, only_y=True)
#             im_Gen_proc = bgr2ycbcr(im_Gen, only_y=True)
#         else:
#             im_GT_proc = im_GT
#             im_Gen_proc = im_Gen
#
#         # crop_border
#         if crop_border > 0:
#             h, w = im_GT_proc.shape[:2]
#             if h > 2*crop_border and w > 2*crop_border:
#                 if im_GT_proc.ndim == 3:
#                     im_GT_proc = im_GT_proc[crop_border:-crop_border, crop_border:-crop_border, :]
#                     im_Gen_proc = im_Gen_proc[crop_border:-crop_border, crop_border:-crop_border, :]
#                 else:
#                     im_GT_proc = im_GT_proc[crop_border:-crop_border, crop_border:-crop_border]
#                     im_Gen_proc = im_Gen_proc[crop_border:-crop_border, crop_border:-crop_border]
#
#         # 计算 PSNR (放大到 0..255)
#         psnr_value = calculate_psnr(im_GT_proc * 255.0, im_Gen_proc * 255.0)
#         total_psnr += psnr_value
#         processed += 1
#
#         epochfile.write(f"{GT_imgName},{round(psnr_value, 6)}\n")
#         if idx % 50 == 0:
#             print(f"=== PSNR is processing {idx:>3d}-th image ===")
#
#     endtime = datetime.datetime.now()
#     if processed == 0:
#         avg_psnr = 0.0
#         print("No images processed. Avg PSNR = 0.0")
#     else:
#         avg_psnr = total_psnr / processed
#         print(f"======================= Complete the PSNR test of {processed:>3d} images, "
#               f"take {(endtime - starttime).seconds} seconds =======================")
#         print(f"Average PSNR: {avg_psnr:.4f} dB")
#
#     epochfile.write(f"Average,{round(avg_psnr, 6)}\n")
#     epochfile.close()
#     totalfile.write(f"{epoch},{round(avg_psnr, 6)}\n")
#     totalfile.close()
#
#     return avg_psnr

#------------------------------------------------------------------------------------------------------
def calc_psnr(folder_Gen, folder_GT, result_save_path, epoch):

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    PSNR, total_psnr, avg_psnr = 0.0, 0.0, 0.0
    epoch_result = result_save_path + 'PSNR_epoch_' + str(epoch) + '.csv'
    epochfile = open(epoch_result, 'w')
    epochfile.write('image_name' + ','+ 'psnr' + '\n')

    total_result = result_save_path + 'PSNR_total_results_epoch_avgpsnr.csv'
    totalfile = open(total_result, 'a+')

    crop_border = 4
    test_Y = True  # True: test Y channel only; False: test RGB channels

    img_list = sorted(glob.glob(folder_Gen + '/*'))
    print(len(img_list))
    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    starttime = datetime.datetime.now()

    for i, img_path in enumerate(img_list):
        im_Gen = cv2.imread(img_path) / 255.
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        imgName, _, _ = base_name.rsplit('_', 2)
        # print(base_name)
        # print(imgName)
        GT_imgName = imgName + '.png'
        # print(GT_imgName)
        im_GT = cv2.imread(os.path.join(folder_GT, GT_imgName)) / 255.

        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if im_GT_in.ndim == 3:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_GT_in.ndim == 2:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)

        # print("=== PSNR is {:.4f} of {:>3d}-th img ===".format(PSNR, i))

        epochfile.write(GT_imgName + ',' + str(round(PSNR, 6)) + '\n')

        total_psnr += PSNR
        if i % 50 == 0:
            print("=== PSNR is processing {:>3d}-th image ===".format(i))

    endtime = datetime.datetime.now()
    print("======================= Complete the PSNR test of {:>3d} images, take {} seconds ======================= ".format(i+1, (endtime - starttime).seconds))
    avg_psnr = total_psnr / i
    epochfile.write('Average' + ',' + str(round(avg_psnr, 6)) + '\n')
    epochfile.close()
    totalfile.write(str(epoch) + ',' + str(round(avg_psnr, 6)) + '\n')
    totalfile.close()
    return avg_psnr



def calculate_psnr(img1, img2, data_range=255):
    # img1 and img2 have range [0, 255]
    # 确保两张图像大小相同
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2, dtype=np.float64)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 10 * np.log10((data_range ** 2)/ mse)
#
#
def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
#
#
# def calculate_ssim(img1, img2):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')
#
#