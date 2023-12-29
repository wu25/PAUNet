# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
#
# from basicsr.metrics.metric_util import reorder_image, to_y_channel
# from basicsr.utils.color_util import rgb2ycbcr_pt
# from basicsr.utils.registry import METRIC_REGISTRY
#
#
# @METRIC_REGISTRY.register()
# def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=True, **kwargs):
#     """Calculate PSNR (Peak Signal-to-Noise Ratio).
#
#     Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#
#     Args:
#         img (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#         crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
#
#     Returns:
#         float: PSNR result.
#     """
#
#     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
#     img = reorder_image(img, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#
#     if crop_border != 0:
#         img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
#
#     if test_y_channel:
#         img = to_y_channel(img)
#         img2 = to_y_channel(img2)
#
#     img = img.astype(np.float64)
#     img2 = img2.astype(np.float64)
#
#     mse = np.mean((img - img2)**2)
#     if mse == 0:
#         return float('inf')
#     return 10. * np.log10(255. * 255. / mse)
#
#
# @METRIC_REGISTRY.register()
# def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
#     """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).
#
#     Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#
#     Args:
#         img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
#         img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
#         crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
#
#     Returns:
#         float: PSNR result.
#     """
#
#     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#
#     if crop_border != 0:
#         img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
#         img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
#
#     if test_y_channel:
#         img = rgb2ycbcr_pt(img, y_only=True)
#         img2 = rgb2ycbcr_pt(img2, y_only=True)
#
#     img = img.to(torch.float64)
#     img2 = img2.to(torch.float64)
#
#     mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
#     return 10. * torch.log10(1. / (mse + 1e-8))
#
#
# @METRIC_REGISTRY.register()
# # def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
# #     """Calculate SSIM (structural similarity).
# #
# #     ``Paper: Image quality assessment: From error visibility to structural similarity``
# #
# #     The results are the same as that of the official released MATLAB code in
# #     https://ece.uwaterloo.ca/~z70wang/research/ssim/.
# #
# #     For three-channel images, SSIM is calculated for each channel and then
# #     averaged.
# #
# #     Args:
# #         img (ndarray): Images with range [0, 255].
# #         img2 (ndarray): Images with range [0, 255].
# #         crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
# #         input_order (str): Whether the input order is 'HWC' or 'CHW'.
# #             Default: 'HWC'.
# #         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
# #
# #     Returns:
# #         float: SSIM result.
# #     """
# #
# #     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
# #     if input_order not in ['HWC', 'CHW']:
# #         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
# #     img = reorder_image(img, input_order=input_order)
# #     img2 = reorder_image(img2, input_order=input_order)
# #
# #     if crop_border != 0:
# #         img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
# #         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
# #
# #     if test_y_channel:
# #         img = to_y_channel(img)
# #         img2 = to_y_channel(img2)
# #
# #     img = img.astype(np.float64)
# #     img2 = img2.astype(np.float64)
# #
# #     ssims = []
# #     for i in range(img.shape[2]):
# #         ssims.append(_ssim(img[..., i], img2[..., i]))
# #     return np.array(ssims).mean()
# def _3d_gaussian_calculator(img, conv3d):
#     out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
#     return out
# def _generate_3d_gaussian_kernel():
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     kernel_3 = cv2.getGaussianKernel(11, 1.5)
#     kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
#     conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
#     conv3d.weight.requires_grad = False
#     conv3d.weight[0, 0, :, :, :] = kernel
#     return conv3d
# def _ssim_3d(img1, img2, max_value):
#     assert len(img1.shape) == 3 and len(img2.shape) == 3
#     """Calculate SSIM (structural similarity) for one channel images.
#
#     It is called by func:`calculate_ssim`.
#
#     Args:
#         img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
#         img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
#
#     Returns:
#         float: ssim result.
#     """
#     C1 = (0.01 * max_value) ** 2
#     C2 = (0.03 * max_value) ** 2
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#
#     kernel = _generate_3d_gaussian_kernel().cuda()
#
#     img1 = torch.tensor(img1).float().cuda()
#     img2 = torch.tensor(img2).float().cuda()
#
#
#     mu1 = _3d_gaussian_calculator(img1, kernel)
#     mu2 = _3d_gaussian_calculator(img2, kernel)
#
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
#     sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
#     sigma12 = _3d_gaussian_calculator(img1*img2, kernel) - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) *
#                 (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                        (sigma1_sq + sigma2_sq + C2))
#     return float(ssim_map.mean())
# def _ssim_cly(img1, img2):
#     assert len(img1.shape) == 2 and len(img2.shape) == 2
#     """Calculate SSIM (structural similarity) for one channel images.
#
#     It is called by func:`calculate_ssim`.
#
#     Args:
#         img1 (ndarray): Images with range [0, 255] with order 'HWC'.
#         img2 (ndarray): Images with range [0, 255] with order 'HWC'.
#
#     Returns:
#         float: ssim result.
#     """
#
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     # print(kernel)
#     window = np.outer(kernel, kernel.transpose())
#
#     bt = cv2.BORDER_REPLICATE
#
#     mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
#     mu2 = cv2.filter2D(img2, -1, window,borderType=bt)
#
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=bt) - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=bt) - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) *
#                 (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                        (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
# def calculate_ssim(img1,
#                    img2,
#                    crop_border,
#                    input_order='HWC',
#                    test_y_channel=False):
#     """Calculate SSIM (structural similarity).
#
#     Ref:
#     Image quality assessment: From error visibility to structural similarity
#
#     The results are the same as that of the official released MATLAB code in
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/.
#
#     For three-channel images, SSIM is calculated for each channel and then
#     averaged.
#
#     Args:
#         img1 (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#         crop_border (int): Cropped pixels in each edge of an image. These
#             pixels are not involved in the SSIM calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'.
#             Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
#
#     Returns:
#         float: ssim result.
#     """
#
#     assert img1.shape == img2.shape, (
#         f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(
#             f'Wrong input_order {input_order}. Supported input_orders are '
#             '"HWC" and "CHW"')
#
#     if type(img1) == torch.Tensor:
#         if len(img1.shape) == 4:
#             img1 = img1.squeeze(0)
#         img1 = img1.detach().cpu().numpy().transpose(1,2,0)
#     if type(img2) == torch.Tensor:
#         if len(img2.shape) == 4:
#             img2 = img2.squeeze(0)
#         img2 = img2.detach().cpu().numpy().transpose(1,2,0)
#
#     img1 = reorder_image(img1, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#
#     if crop_border != 0:
#         img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
#
#     if test_y_channel:
#         img1 = to_y_channel(img1)
#         img2 = to_y_channel(img2)
#         return _ssim_cly(img1[..., 0], img2[..., 0])
#
#
#     ssims = []
#     # ssims_before = []
#
#     # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
#     # print('.._skimage',
#     #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
#     max_value = 1 if img1.max() <= 1 else 255
#     with torch.no_grad():
#         final_ssim = _ssim_3d(img1, img2, max_value)
#         ssims.append(final_ssim)
#
#     # for i in range(img1.shape[2]):
#     #     ssims_before.append(_ssim(img1, img2))
#
#     # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
#         # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))
#
#     return np.array(ssims).mean()
#
#
# @METRIC_REGISTRY.register()
# def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
#     """Calculate SSIM (structural similarity) (PyTorch version).
#
#     ``Paper: Image quality assessment: From error visibility to structural similarity``
#
#     The results are the same as that of the official released MATLAB code in
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/.
#
#     For three-channel images, SSIM is calculated for each channel and then
#     averaged.
#
#     Args:
#         img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
#         img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
#         crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
#
#     Returns:
#         float: SSIM result.
#     """
#
#     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#
#     if crop_border != 0:
#         img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
#         img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
#
#     if test_y_channel:
#         img = rgb2ycbcr_pt(img, y_only=True)
#         img2 = rgb2ycbcr_pt(img2, y_only=True)
#
#     img = img.to(torch.float64)
#     img2 = img2.to(torch.float64)
#
#     ssim = _ssim_pth(img * 255., img2 * 255.)
#     return ssim
#
#
# def _ssim(img, img2):
#     """Calculate SSIM (structural similarity) for one channel images.
#
#     It is called by func:`calculate_ssim`.
#
#     Args:
#         img (ndarray): Images with range [0, 255] with order 'HWC'.
#         img2 (ndarray): Images with range [0, 255] with order 'HWC'.
#
#     Returns:
#         float: SSIM result.
#     """
#
#     c1 = (0.01 * 255)**2
#     c2 = (0.03 * 255)**2
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
#     return ssim_map.mean()
#
#
# def _ssim_pth(img, img2):
#     """Calculate SSIM (structural similarity) (PyTorch version).
#
#     It is called by func:`calculate_ssim_pt`.
#
#     Args:
#         img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
#         img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
#
#     Returns:
#         float: SSIM result.
#     """
#     c1 = (0.01 * 255)**2
#     c2 = (0.03 * 255)**2
#
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)
#
#     mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
#     mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
#     sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2
#
#     cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
#     ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
#     return ssim_map.mean([1, 2, 3])








import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY


# @METRIC_REGISTRY.register()
# def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
#     """Calculate PSNR (Peak Signal-to-Noise Ratio).
#
#     Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#
#     Args:
#         img (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#         crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
#
#     Returns:
#         float: PSNR result.
#     """
#
#     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
#     img = reorder_image(img, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#
#     if crop_border != 0:
#         img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
#
#     if test_y_channel:
#         img = to_y_channel(img)
#         img2 = to_y_channel(img2)
#
#     img = img.astype(np.float64)
#     img2 = img2.astype(np.float64)
#
#     mse = np.mean((img - img2)**2)
#     if mse == 0:
#         return float('inf')
#     return 10. * np.log10(255. * 255. / mse)
#
#
# @METRIC_REGISTRY.register()
# def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
#     """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).
#
#     Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#
#     Args:
#         img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
#         img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
#         crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
#
#     Returns:
#         float: PSNR result.
#     """
#
#     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#
#     if crop_border != 0:
#         img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
#         img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
#
#     if test_y_channel:
#         img = rgb2ycbcr_pt(img, y_only=True)
#         img2 = rgb2ycbcr_pt(img2, y_only=True)
#
#     img = img.to(torch.float64)
#     img2 = img2.to(torch.float64)
#
#     mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
#     return 10. * torch.log10(1. / (mse + 1e-8))
@METRIC_REGISTRY.register()

def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))



@METRIC_REGISTRY.register()
# def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
#     """Calculate SSIM (structural similarity).
#
#     ``Paper: Image quality assessment: From error visibility to structural similarity``
#
#     The results are the same as that of the official released MATLAB code in
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/.
#
#     For three-channel images, SSIM is calculated for each channel and then
#     averaged.
#
#     Args:
#         img (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#         crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'.
#             Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
#
#     Returns:
#         float: SSIM result.
#     """
#
#     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
#     img = reorder_image(img, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#
#     if crop_border != 0:
#         img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
#
#     if test_y_channel:
#         img = to_y_channel(img)
#         img2 = to_y_channel(img2)
#
#     img = img.astype(np.float64)
#     img2 = img2.astype(np.float64)
#
#     ssims = []
#     for i in range(img.shape[2]):
#         ssims.append(_ssim(img[..., i], img2[..., i]))
#     return np.array(ssims).mean()
def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out
def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d
def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()


    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1*img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())
def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window,borderType=bt)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        return _ssim_cly(img1[..., 0], img2[..., 0])


    ssims = []
    # ssims_before = []

    # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
    # print('.._skimage',
    #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
    max_value = 1 if img1.max() <= 1 else 255
    with torch.no_grad():
        final_ssim = _ssim_3d(img1, img2, max_value)
        ssims.append(final_ssim)

    # for i in range(img1.shape[2]):
    #     ssims_before.append(_ssim(img1, img2))

    # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
        # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

    return np.array(ssims).mean()


@METRIC_REGISTRY.register()
def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity) (PyTorch version).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    It is called by func:`calculate_ssim_pt`.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])
