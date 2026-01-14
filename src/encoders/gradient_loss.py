import torch
import torch.nn.functional as F


def sobel_gradients(img):
    B, C, H, W = img.shape

    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=img.device).view(1, 1, 3, 3)

    kx = sobel_x.repeat(C, 1, 1, 1)  # repeat c many times with c being channel 3 channels 3 kernels
    ky = sobel_y.repeat(C, 1, 1, 1)

    gx = F.conv2d(img, kx, padding=1, groups=C)  # c makes sure channels dont mix 1 kernel 1 channel
    gy = F.conv2d(img, ky, padding=1, groups=C)
    return gx, gy

def sobel_gradient_loss(pred, target):
    pred_gx, pred_gy = sobel_gradients(pred)
    targ_gx, targ_gy = sobel_gradients(target)
    return (pred_gx - targ_gx).abs().mean() + (pred_gy - targ_gy).abs().mean()
