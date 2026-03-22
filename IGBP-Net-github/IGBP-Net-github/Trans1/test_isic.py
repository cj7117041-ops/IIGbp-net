import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from utils.dataloader import test_dataset
import imageio
import glob
from utils.dataloaders import get_dataloaders
from lib.MoXing import IGBP

# ==========================================
def calculate_bootstrap_ci(data, n_bootstraps=1000, ci_level=0.95):
    """
    Calculate the 95% confidence interval for the given data list
    """
    data = np.array(data)
    n = len(data)
    bootstrapped_means = []

    np.random.seed(42)

    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = 1.0 - ci_level
    lower_bound = np.percentile(bootstrapped_means, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrapped_means, 100 * (1 - alpha / 2))

    mean_val = np.mean(data)
    return mean_val, lower_bound, upper_bound
# ==========================================

def mean_rec_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes)

    smooth = .001
    rec = (intersection + smooth) / (mask_sum + smooth)
    if intersection == 0:
        rec = 0
    return rec

def mean_prec_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    prec = (intersection + smooth) / (mask_sum + smooth)
    if intersection == 0:
        prec = 0
    return prec

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    if intersection == 0:
        iou = 0
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    if intersection == 0:
        dice=0
    return dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='.pth')
    parser.add_argument('--save_path', type=str,
                        default='\weights',
                        help='path to save inference segmentation')
    parser.add_argument('--dataset', type=str,
                        default='',
                        help='name of dataset')  # 'Kvasir' 'CVC_ClinicDB'
    parser.add_argument('--root', type=str,
                        default='dataset/data',
                        help='path to dataset')

    opt = parser.parse_args()

    if opt.dataset == "--dataset":
        img_path = opt.root + "../test/images/*"
        input_paths = glob.glob(img_path)  # sorted(glob.glob(img_path))

        depth_path = opt.root + "../test/masks/*"
        target_paths = glob.glob(depth_path)  # sorted(glob.glob(depth_path))
    elif opt.dataset == "--dataset":
        img_path = opt.root + "../test/images/*"
        input_paths = glob.glob(img_path)  # sorted(glob.glob(img_path))
        depth_path = opt.root + "../test/masks/*"
        target_paths = glob.glob(depth_path)  # sorted(glob.glob(depth_path))
    elif opt.dataset == "--dataset":
        img_path = opt.root + "../images/*"
        input_paths = glob.glob(img_path)  # sorted(glob.glob(img_path))
        depth_path = opt.root + "../masks/*"
        target_paths = glob.glob(depth_path)  # sorted(glob.glob(depth_path))
    elif opt.dataset == "--dataset":
        img_path = opt.root + "../images/*"
        input_paths = glob.glob(img_path)  # sorted(glob.glob(img_path))
        depth_path = opt.root + "../masks/*"
        target_paths = glob.glob(depth_path)  # sorted(glob.glob(depth_path))

    model = IGBP().cuda()

    model.load_state_dict(torch.load(opt.ckpt_path, map_location='cuda:0'))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    print('evaluating model: ', opt.ckpt_path)

    test_loader = get_dataloaders(
        input_paths, target_paths, batch_size=1, if_train=False
    )

    dice_bank = []
    iou_bank = []
    prec_bank = []
    rec_bank = []
    acc_bank = []

    for i, pack in enumerate(test_loader):
        image, gt = pack

        image = image.cuda()

        with torch.no_grad():
             res, _, _, _, boundary= model(image)


        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 1*(res > 0.5)
        gt = gt.data.cpu().numpy().squeeze()
        gt = 1 * (gt > 0.5)


        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        prec = mean_prec_np(gt, res)
        rec = mean_rec_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)
        prec_bank.append(prec)
        rec_bank.append(rec)

   print('Raw Mean -> Dice: {:.4f}, IoU: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, Acc: {:.4f}'.
          format(np.mean(dice_bank), np.mean(iou_bank), np.mean(prec_bank), np.mean(rec_bank), np.mean(acc_bank)))
   print("\n" + "=" * 50)

    # Calculate the confidence interval of Dice
    mean_dice, ci_lower_dice, ci_upper_dice = calculate_bootstrap_ci(dice_bank)
    margin_dice = (ci_upper_dice - mean_dice) * 100

    # Calculate the confidence interval of IOU
    mean_iou, ci_lower_iou, ci_upper_iou = calculate_bootstrap_ci(iou_bank)
    margin_iou = (ci_upper_iou - mean_iou) * 100

    # Calculate the confidence interval of Prec
    mean_prec, ci_lower_prec, ci_upper_prec = calculate_bootstrap_ci(prec_bank)
    margin_prec = (ci_upper_prec - mean_prec) * 100

    # Calculate the confidence interval of rec
    mean_rec, ci_lower_rec, ci_upper_rec = calculate_bootstrap_ci(rec_bank)
    margin_rec = (ci_upper_rec - mean_rec) * 100

    print(f"Dice: {mean_dice * 100:.2f} ± {margin_dice:.2f}")
    print(f"IoU:  {mean_iou * 100:.2f} ± {margin_iou:.2f}")
    print(f"prec: {mean_prec * 100:.2f} ± {margin_prec:.2f}")
    print(f"rec:  {mean_rec * 100:.2f} ± {margin_rec:.2f}")
