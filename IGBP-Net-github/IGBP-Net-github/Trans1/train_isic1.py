import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.MoXing import IGBP
from utils.dataloader1 import get_dataloaders
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
from test_isic import mean_dice_np, mean_iou_np, mean_prec_np, mean_rec_np
import os
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0")


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  # 增加边界区域和目标的权重
    # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + 2 * mask
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def structure_loss2(pred, mask):
    weit = 1 + 5 * mask  # 增加边界区域和目标的权重
    # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + 2 * mask
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, val_loader, test_loader, model, optimizer, epoch, best_loss):
    model.train()
    # size_rates = [0.75, 1, 1.25]  # ##
    size_rates = [1]
    loss_record = AvgMeter()
    accum = 0
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:  # ##
            # ---- data prepare ----
            images, gts, boundary_gt = pack


            images = Variable(images).to(device)  # ############
            gts = Variable(gts).to(device)
            boundary_gt = Variable(boundary_gt).to(device)

            # rescale
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:  # ##
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)  # ##
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)  # ##
                boundary_gt = F.interpolate(boundary_gt, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_1, lateral_map_2, lateral_map_3, saliency_map, boundary = model(images)

            loss2 = structure_loss(saliency_map, gts)
            loss3 = structure_loss(lateral_map_2, gts)
            loss4 = structure_loss(lateral_map_3, gts)
            loss1 = structure_loss(lateral_map_1, gts)
            loss6 = structure_loss2(boundary, boundary_gt)
            loss = loss1 + 0.4 * loss2 + 0.8 * loss3 + 0.6 * loss4 + 2 * loss6

            # ---- backward ----
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # ---- recording loss ----
            if rate == 1:  # ##
                loss_record.update(loss.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))

    save_path = 'weights/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch % 5 == 0) and epoch >= 20:
    # if (epoch % 4 == 0) and epoch >= 20:
    # if epoch >= 0:
        print("test dataset1 ......")
        meanloss1, iou1 = test(val_loader, model, opt.test_path)
        print("test dataset2......")
        meanloss2, iou2 = test(test_loader, model, opt.test_path)
        if iou1 > 0.8800 and iou2 > 0.8900:
            print('better iou: ', iou1, iou2)
            torch.save(model.state_dict(), save_path + 'TransFuse-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'TransFuse-%d.pth' % epoch)
        if (meanloss1 + meanloss2) / 2 < best_loss:
            print('new best loss: ', meanloss1, meanloss2)        #
            best_loss = (meanloss1 + meanloss2) / 2
            torch.save(model.state_dict(), save_path + 'TransFuse-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'TransFuse-%d.pth' % epoch)

    return best_loss


def test(val_loader, model, path):
    model.eval()
    mean_loss = []

    for s in ['val']:


        dice_bank = []
        iou_bank = []
        prec_bank = []
        rec_bank = []
        loss_bank = []
        acc_bank = []
        MAE_bank = []
        for i, (image, gt, boundary_gt) in enumerate(val_loader):
            # for i in range(test_loader.size):
            #     image, gt = test_loader.load_data()
            # image = image.cuda()
            image = image.to(device)  # ############

            with torch.no_grad():
                res1, res2, res3, res4, boundary = model(image)
                # res1, _, _, _, _, _, boundary = model(image)

            loss = structure_loss(res1, torch.tensor(gt).to(device))  # ############

            res1 = res1.sigmoid().data.cpu().numpy().squeeze()

            gt = gt.data.cpu().numpy().squeeze()
            gt = 1. * (gt > 0.5)  # 会使得大于0.5的像素点为1，其余的为0.
            res1 = 1. * (res1 > 0.5)

            MAE = np.abs(gt - res1).mean()
            dice = mean_dice_np(gt, res1)
            iou = mean_iou_np(gt, res1)
            prec = mean_prec_np(gt, res1)
            rec = mean_rec_np(gt, res1)
            acc = np.sum(res1 == gt) / (res1.shape[0] * res1.shape[1])

            loss_bank.append(loss.item())
            dice_bank.append(dice)
            iou_bank.append(iou)
            prec_bank.append(prec)
            rec_bank.append(rec)
            acc_bank.append(acc)
            MAE_bank.append(MAE)


        print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, MAE: {:.4f}, Acc: {:.4f}'.
              format(s, np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), np.mean(prec_bank),
                     np.mean(rec_bank), np.mean(MAE_bank), np.mean(acc_bank)))

        mean_loss.append(np.mean(loss_bank))

    return mean_loss[0], np.mean(iou_bank)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1, help='epoch number')  # 100 25
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 5e-5 1e-4 4e-4 3e-4 5e-4 8e-5
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='dataset', help='path to train dataset')  #data/dataset
    parser.add_argument('--root', type=str,
                        default='root/', help='path to dataset')
    parser.add_argument('--dataset', type=str,
                        default='dataset1/2',
                        help='name of dataset')
    parser.add_argument('--test_path', type=str,
                        default='data', help='path to train1 dataset')
    parser.add_argument('--train_save', type=str, default='')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    opt = parser.parse_args()
    model = IGBP().to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))

    if opt.dataset == "dataset1":
        img_path = opt.root + "dataset1/train/images/*"
        input_paths = glob.glob(img_path)  # sorted(glob.glob(img_path))

        depth_path = opt.root + "dataset1/train/masks/*"
        target_paths = glob.glob(depth_path)  # sorted(glob.glob(depth_path))
        image_val_path = opt.root + "dataset1/val/images/*"
        val_image = glob.glob(image_val_path)
        mask_val_path = opt.root + "dataset1/val/masks/*"
        val_mask = glob.glob(mask_val_path)
        image_test_path = opt.root + "dataset1/test/images/*"
        test_image = glob.glob(image_test_path)
        mask_test_path = opt.root + "dataset1/test/masks/*"
        test_mask = glob.glob(mask_test_path)
    elif opt.dataset == "dataset2":
        img_path = opt.root + "dataset2/train/images/*"
        input_paths = glob.glob(img_path)  # sorted(glob.glob(img_path))
        depth_path = opt.root + "dataset2/train/masks/*"
        target_paths = glob.glob(depth_path)  # sorted(glob.glob(depth_path))
        boundary_path = opt.root + "dataset2/train/boundary_dilation/*"
        boundary_paths = glob.glob(boundary_path)  # sorted(glob.glob(depth_path))

        image_test_path1 = opt.root + "dataset2/test/images/*"
        test_image1 = glob.glob(image_test_path1)
        mask_test_path1 = opt.root + "dataset2/test/masks/*"
        test_mask1 = glob.glob(mask_test_path1)
        boundary_path1 = opt.root + "dataset2/boundary_dilation/*"
        boundary_paths1 = glob.glob(boundary_path1)  # sorted(glob.glob(depth_path))

        image_test_path2 = opt.root + "dataset2/test/images/*"
        test_image2 = glob.glob(image_test_path2)
        mask_test_path2 = opt.root + "dataset2/test/masks/*"
        test_mask2 = glob.glob(mask_test_path2)
        boundary_path2 = opt.root + "dataset2/boundary_dilation/*"
        boundary_paths2 = glob.glob(boundary_path2)  # sorted(glob.glob(depth_path))

    train_loader = get_dataloaders(
        input_paths, target_paths, boundary_paths, batch_size=opt.batchsize, if_train=True
    )
    test1_loader = get_dataloaders(
        test_image1, test_mask1, boundary_paths1, batch_size=1, if_train=False
    )
    test2_loader = get_dataloaders(
        test_image2, test_mask2, boundary_paths2, batch_size=1, if_train=False
    )
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    best_loss = 1e5
    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, test1_loader, test2_loader, model, optimizer, epoch, best_loss)


