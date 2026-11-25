import os
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
import cv2
import glob
import datetime
from PIL import Image

# ------- 复用的工具函数 -------
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
#     if img.dtype == np.uint8:
#         img_f = img.astype(np.float32) / 255.0
#     else:
#         img_f = img.astype(np.float32)
#         if img_f.max() > 1.5:
#             img_f = img_f / 255.0
#
#     R, G, B = img_f[..., 0], img_f[..., 1], img_f[..., 2]
#     Y = 0.299 * R + 0.587 * G + 0.114 * B
#     if only_y:
#         return Y
#     else:
#         Cb = 0.5 * (B - Y) / (1 - 0.114) + 0.5
#         Cr = 0.5 * (R - Y) / (1 - 0.299) + 0.5
#         return np.stack([Y, Cb, Cr], axis=-1)
#
#
# # ---------------- ECC 对齐函数 ----------------
# def ecc_align(src_rgb_f01, ref_rgb_f01, warp_mode='affine', number_of_iterations=500, termination_eps=1e-6):
#
#     def to_gray_float32(img_f01):
#         if img_f01.dtype != np.float32:
#             img_f01 = img_f01.astype(np.float32)
#         gray = cv2.cvtColor((np.clip(img_f01, 0, 1)*255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#         return gray.astype(np.float32) / 255.0
#
#     src_h, src_w = src_rgb_f01.shape[:2]
#     ref_h, ref_w = ref_rgb_f01.shape[:2]
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
#     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
#                 number_of_iterations, termination_eps)
#     (cc, warp_matrix) = cv2.findTransformECC(ref_gray, src_gray,
#                                              warp_matrix, cv_warp_mode, criteria,
#                                              inputMask=None, gaussFiltSize=1)
#
#     if cv_warp_mode == cv2.MOTION_HOMOGRAPHY:
#         aligned = cv2.warpPerspective((np.clip(src_rgb_f01, 0, 1)*255.0).astype(np.uint8),
#                                       warp_matrix, (ref_w, ref_h),
#                                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#     else:
#         aligned = cv2.warpAffine((np.clip(src_rgb_f01, 0, 1)*255.0).astype(np.uint8),
#                                  warp_matrix, (ref_w, ref_h),
#                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#     aligned_f01 = aligned.astype(np.float32) / 255.0
#     return aligned_f01
#
#
# # ---------------- 主函数：带 ECC 的 calc_ssim ----------------
# def calc_ssim(folder_Gen, folder_GT, result_save_path, epoch,
#               on_mismatch='crop', use_ecc=True, ecc_warp='affine'):
#     """
#     增强版 calc_ssim：
#       - use_ecc: 是否尝试在对齐前用 ECC 把 GT 对齐到 Gen（默认 False）
#       - ecc_warp: 'affine' 或 'homography'
#       - on_mismatch: 'raise'/'resize'/'crop'
#     """
#     if not os.path.exists(result_save_path):
#         os.makedirs(result_save_path)
#
#     total_ssim = 0.0
#     processed = 0
#     epoch_result = os.path.join(result_save_path, f'SSIM_epoch_{epoch}.csv')
#     total_result = os.path.join(result_save_path, 'SSIM_total_results_epoch_avgssim.csv')
#
#     epochfile = open(epoch_result, 'w')
#     epochfile.write('image_name,ssim\n')
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
#         # --- ECC 对齐 ---
#         if use_ecc:
#             try:
#                 if im_GT.shape[:2] != im_Gen.shape[:2]:
#                     im_GT_resz = resize_lanczos_float01(im_GT, im_Gen.shape[1], im_Gen.shape[0])
#                 else:
#                     im_GT_resz = im_GT
#                 im_GT_aligned = ecc_align(im_GT_resz, im_Gen, warp_mode=ecc_warp)
#                 im_GT = im_GT_aligned
#                 if not mismatch_reported:
#                     print(f"[Info] ECC alignment succeeded for {GT_imgName} (warp={ecc_warp}).")
#                     mismatch_reported = True
#             except Exception as e:
#                 print(f"[Warning] ECC alignment failed for {GT_imgName}: {e}. Falling back to normal handling.")
#
#         # --- 尺寸不匹配 ---
#         if im_Gen.shape != im_GT.shape:
#             h_g, w_g = im_Gen.shape[:2]
#             h_t, w_t = im_GT.shape[:2]
#
#             if on_mismatch == 'raise':
#                 raise ValueError(f"Shape mismatch: {im_Gen.shape} vs {im_GT.shape}")
#             elif on_mismatch == 'resize':
#                 im_Gen = resize_lanczos_float01(im_Gen, w_t, h_t)
#                 if not mismatch_reported:
#                     print(f"[Info] Resized Gen ({w_g}x{h_g}) -> GT ({w_t}x{h_t}) using LANCZOS")
#                     mismatch_reported = True
#             elif on_mismatch == 'crop':
#                 min_h, min_w = min(h_g, h_t), min(w_g, w_t)
#                 try:
#                     if (h_g, w_g) != (min_h, min_w):
#                         im_Gen = center_crop(im_Gen, min_h, min_w)
#                     if (h_t, w_t) != (min_h, min_w):
#                         im_GT = center_crop(im_GT, min_h, min_w)
#                     if not mismatch_reported:
#                         print(f"[Info] Center-cropped to ({min_h}x{min_w}) for {GT_imgName}")
#                         mismatch_reported = True
#                 except Exception:
#                     im_Gen = resize_lanczos_float01(im_Gen, w_t, h_t)
#                     if not mismatch_reported:
#                         print(f"[Info] Center-crop failed; resized Gen -> GT using LANCZOS")
#                         mismatch_reported = True
#         # crop_border
#         if crop_border > 0:
#             h, w = im_GT.shape[:2]
#             if h > 2 * crop_border and w > 2 * crop_border:
#                 if im_GT.ndim == 3:
#                     im_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border, :]
#                     im_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border, :]
#                 else:
#                     im_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border]
#                     im_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border]
#
#         # --- Y通道 ---
#         if test_Y:
#             im_GT_proc = bgr2ycbcr(im_GT, only_y=True)
#             im_Gen_proc = bgr2ycbcr(im_Gen, only_y=True)
#         else:
#             im_GT_proc = im_GT
#             im_Gen_proc = im_Gen
#
#         # --- 计算 SSIM ---
#         try:
#             if test_Y:
#                 ssim_value = ssim_skimage(im_GT_proc, im_Gen_proc, data_range=1.0)
#             else:
#                 ssim_value = ssim_skimage(im_GT_proc, im_Gen_proc, channel_axis=-1, data_range=1.0)
#         except TypeError:
#             ssim_value = ssim_skimage(im_GT_proc, im_Gen_proc, multichannel=True, data_range=1.0)
#
#         total_ssim += ssim_value
#         processed += 1
#         epochfile.write(f"{GT_imgName},{round(ssim_value, 6)}\n")
#
#         if idx % 50 == 0:
#             print(f"=== SSIM is processing {idx:>3d}-th image ===")
#
#     endtime = datetime.datetime.now()
#     if processed == 0:
#         avg_ssim = 0.0
#         print("No images processed. Avg SSIM = 0.0")
#     else:
#         avg_ssim = total_ssim / processed
#         print(f"======================= Complete the SSIM test of {processed:>3d} images, "
#               f"take {(endtime - starttime).seconds} seconds =======================")
#         print(f"Average SSIM: {avg_ssim:.6f}")
#
#     epochfile.write(f"Average,{round(avg_ssim, 6)}\n")
#     epochfile.close()
#     totalfile.write(f"{epoch},{round(avg_ssim, 6)}\n")
#     totalfile.close()
#
#     return avg_ssim
# ------------------------------------------------------------------------------------------------------------------
def calc_ssim(folder_Gen, folder_GT, result_save_path, epoch):

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    SSIM, total_ssim, avg_ssim = 0.0, 0.0, 0.0
    epoch_result = result_save_path + 'SSIM_epoch_' + str(epoch) + '.csv'
    epochfile = open(epoch_result, 'w')
    epochfile.write('image_name' + ','+ 'ssim' + '\n')

    total_result = result_save_path + 'SSIM_total_results_epoch_avgssim.csv'
    totalfile = open(total_result, 'a+')

    crop_border = 4
    test_Y = True  # True: test Y channel only; False: test RGB channels

    img_list = sorted(glob.glob(folder_Gen + '/*'))
    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    starttime = datetime.datetime.now()

    for i, img_path in enumerate(img_list):
        im_Gen = cv2.imread(img_path) / 255.

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        imgName, _, _ = base_name.rsplit('_', 2)
        GT_imgName = imgName + '.png'
        # GT_imgName = base_name + '.png'
        im_GT = cv2.imread(os.path.join(folder_GT, GT_imgName)) / 255.

        if len(im_GT.shape) == 2:  # GT是灰度图，转为3通道   #后加
            im_GT = cv2.cvtColor(im_GT, cv2.COLOR_GRAY2BGR)
        if len(im_Gen.shape) == 2:  # 生成图是灰度图，转为3通道
            im_Gen = cv2.cvtColor(im_Gen, cv2.COLOR_GRAY2BGR)
        # 确保尺寸匹配
        if im_GT.shape != im_Gen.shape:  #后加
        #     # print(f"Resizing {img_path} to match GT size {im_GT.shape}")
            im_Gen = cv2.resize(im_Gen, (im_GT.shape[1], im_GT.shape[0]))

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
        # SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
        SSIM = ssim_skimage(cropped_GT * 255, cropped_Gen * 255, multichannel=True, data_range=255)
        # SSIM = ssim_skimage(cropped_GT * 255, cropped_Gen * 255, win_size=win_size, channel_axis=-1, data_range=255)

        # print("=== SSIM is {:.4f} of {:>3d}-th img ===".format(SSIM, i))

        epochfile.write(GT_imgName + ',' + str(round(SSIM, 6)) + '\n')

        total_ssim += SSIM
        if i % 50 == 0:
            print("=== SSIM is processing {:>3d}-th image ===".format(i))

    endtime = datetime.datetime.now()
    print("======================= Complete the SSIM test of {:>3d} images, take {} seconds ======================= ".format(i+1, (endtime - starttime).seconds))
    avg_ssim = total_ssim / i
    epochfile.write('Average' + ',' + str(round(avg_ssim, 6)) + '\n')
    epochfile.close()
    totalfile.write(str(epoch) + ',' + str(round(avg_ssim, 6)) + '\n')
    totalfile.close()
    return avg_ssim


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    # 确保两张图像大小相同
    if img1.shape != img2.shape:  #后加
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    # 确保两张图是 3 通道
    if len(img1.shape) == 2:  #后加
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:#后加
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    # 确保两张图像大小相同
    if img1.shape != img2.shape: #后加
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

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