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

def mean_spc_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs((1-y_pred) * (1-y_true)), axis=axes)
    mask_sum = np.sum(np.abs(1-y_true), axis=axes)

    smooth = .001
    spc = (intersection + smooth) / (mask_sum + smooth)
    if intersection == 0:
        spc = 0
    return spc

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
        # glob.glob(img_path) 返回该路径模式的所有文件路径，并以列表的形式返回
        # sorted() 用来对返回的文件路径列表进行排序，按照字母顺序对文件路径进行升序排序。排序后的结果作为最终的输出
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
    spc_bank = []
    MAE_bank = []
    acc_bank = []

    for i, pack in enumerate(test_loader):
        image, gt = pack

        image = image.cuda()

        with torch.no_grad():
              # P1, P2 = model(image)
              # res = F.upsample(P1 + P2, size=(352, 352), mode='bilinear', align_corners=False)
              pred = model(image)
        res = pred


        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 1*(res > 0.5)
        gt = gt.data.cpu().numpy().squeeze()
        gt = 1 * (gt > 0.5)


        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        prec = mean_prec_np(gt, res)
        rec = mean_rec_np(gt, res)
        spc = mean_spc_np(gt, res)
        MAE = np.abs(gt - res).mean()
        acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)
        prec_bank.append(prec)
        rec_bank.append(rec)
        spc_bank.append(spc)
        MAE_bank.append(MAE)

    print('Dice: {:.4f}, IoU: {:.4f}, Prec: {:.4f}, Rec: {:.4f},  Spc: {:.4f},  MAE: {:.4f}, Acc: {:.4f}'.
        format(np.mean(dice_bank), np.mean(iou_bank), np.mean(prec_bank), np.mean(rec_bank), np.mean(spc_bank), np.mean(MAE_bank), np.mean(acc_bank)))
