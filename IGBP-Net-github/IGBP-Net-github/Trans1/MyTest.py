# coding=utf-8
import cv2
import numpy as np
import os

# img = cv2.imread("dataset/Kvasir/test/masks/cju414lf2l1lt0801rl3hjllj.jpg", 0)

# img = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯平滑处理原图像降噪。
# canny = cv2.Canny(img, 50, 150)  # 最大最小阈值

# cv2.imshow('Canny', canny)
# cv2.imwrite('gt4.jpg', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cla_path = r'E:\yanghe\TransFuse\TransFuse-main\data\ETIS-LaribPolypDB\masks' + '/'  # 某一类别的子目录
#E:\yanghe\TransFuse\TransFuse-main\dataset\dataset_offic\dataset\TrainDataset\images
#E:\yanghe\TransFuse\TransFuse-main\data\ETIS-LaribPolypDB
masks = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
#
num = len(masks)
for index, mask in enumerate(masks):
    img = cv2.imread(cla_path+mask, 0)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(img, 50, 150)  # 最大最小阈值
    kernel = np.ones((4, 4), np.uint8)
    canny = (cv2.dilate(canny, kernel, iterations=1))

    cv2.imwrite(r'E:\yanghe\TransFuse\TransFuse-main\data\ETIS-LaribPolypDB\boundary_dilation' + '/' + mask, canny)
    # E:\yanghe\TransFuse\TransFuse-main\dataset\kvasir_clinicDB\Kvasir\boundary_dilation
    # image_path.split('/')[-1]


# # import torch
# # import torch.nn as nn
# # image = torch.tensor([[[0., 1., 0.], [1., 2., 3.]],
# #                      [[1., 0., 0.], [4., 5., 6.]],
# #                      [[2., 1., 0.], [1., 1., 1.]]])
# # image1 = torch.tensor([[[0., 1., 0.], [1., 2., 3.], [0., 1., 0.]]])
# # pos_embed = nn.Parameter(torch.zeros(1, 3, 3))
# # # scale = torch.tensor([[1.], [2.], [2.]])
# # # result = scale * image
# # # print(result)
# # # x_b = torch.transpose(image, 0, 1)
# # # print(x_b)
# # # print(pos_embed)
# # # image=image+pos_embed
# # # print(image)
# # import torch.nn.functional as F
# # pred = torch.tensor([0., 1., -5.])
# # image = torch.tensor([0.1, 0.9, 0.2])
# # # wbce = F.binary_cross_entropy_with_logits(pred, image, reduction='none')
# # # print(wbce)
# # # print((torch.tensor([[1.],[0.]])+torch.tensor([[2.],[0.]])).mean())
# # # gt = 1*(image>0.5)
# # # print(gt)
# # # import numpy as np
# # # loss_bank=[1., 2.]
# # # print(np.mean(loss_bank))
# # #
# # # image = (image > 0.5).float()
# # # print(image)
# #
# # image1 = torch.tensor([[[0., 1., 0.], [1., 2., 3.], [0., 1., 0.]],
# #                        [[0., 1., 0.], [1., 2., 3.], [0., 1., 0.]]])
# #
# # image2 = torch.tensor([[[0., 3., 0.], [1., 6., 3.], [0., 2., 0.]],
# #                        [[0., 8., 0.], [1., 7., 3.], [0., 1., 0.]]])
# #
# # print(image2*image1)
#
#
# import cv2
# import numpy as np
# import os
#
# def tif_to_png(image_path,save_path):
#     """
#     :param image_path: *.tif image path
#     :param save_path: *.png image path
#     :return:
#     """
#     img = cv2.imread(image_path,3)
#     # print(img)
#     # print(img.dtype)
#     filename = image_path.split('/')[-1].split('_')[0]
#     # print(filename)
#     save_path = save_path + '/' + filename + '.jpg'
#     cv2.imwrite(save_path,img)
#
# if __name__ == '__main__':
#     root_path = r'data/train1_cvc/images/'
#     save_path = r'data/train1_cvc/images_jpg/'
#     image_files = os.listdir(root_path)
#     for image_file in image_files:
#         tif_to_png(root_path + image_file,save_path)
#
#
# train/val/test = 8/1/1

# import os
# from shutil import copy
# import random
#
# def mkfile(file):
#     if not os.path.exists(file):
#         os.makedirs(file)
#
#
# # 获取data文件夹下所有文件夹名（即需要分类的类名）
# file_path = r'E:\yanghe\TransFuse\TransFuse-main\dataset\Kvasir\val1'
#
# mkfile('dataset/Kvasir/test/images')
# mkfile('dataset/Kvasir/test/masks')
#
# # # 创建 验证集val 文件夹
# mkfile('dataset/Kvasir/val/images')
# mkfile('dataset/Kvasir/val/masks')
#
# # 划分比例，训练集 : 验证集 = 9 : 1
# split_rate = 0.5
#
# # 遍历所有类别的全部图像并按比例分成训练集和验证集
#
# cla_path = file_path + '/' +'images' + '/'  # 某一类别的子目录
# images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
# cla_path_mask = file_path + '/' +'masks' + '/'  # 某一类别的子目录
# masks = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
# num = len(images)
# eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
# for index, image in enumerate(images):
#     # eval_index 中保存验证集val的图像名称
#     if image in eval_index:
#         image_path = cla_path + image
#         new_path = 'dataset/Kvasir/val/images'
#         copy(image_path, new_path)  # 将选中的图像复制到新路径
#
#     # 其余的图像保存在训练集train中
#     else:
#         image_path = cla_path + image
#         new_path = 'dataset/Kvasir/test/images'
#         copy(image_path, new_path)
#
# for index, image in enumerate(masks):
#     # eval_index 中保存验证集val的图像名称
#     if image in eval_index:
#         image_path = cla_path_mask + image
#         new_path = 'dataset/Kvasir/val/masks'
#         copy(image_path, new_path)  # 将选中的图像复制到新路径
#
#     # 其余的图像保存在训练集train中
#     else:
#         image_path = cla_path_mask + image
#         new_path = 'dataset/Kvasir/test/masks'
#         copy(image_path, new_path)
#
#
# print("processing done!")

#
#
# import numpy as np
# from sklearn.model_selection import train_test_split
# def split_ids(len_ids):
#     train_size = int(round((80 / 100) * len_ids))
#     valid_size = int(round((10 / 100) * len_ids))
#     test_size = int(round((10 / 100) * len_ids))
#
#     train_indices, test_indices = train_test_split(
#         np.linspace(0, len_ids - 1, len_ids).astype("int"),
#         test_size=test_size * 2,
#         # random_state=42,
#     )
#     # np.linspace(0, len_ids - 1, len_ids).astype("int")，表示输入的数组。
#     # 数组使用 np.linspace(0, len_ids - 1, len_ids) 生成一个整数类型的等间隔数组，范围从 0 到 len_ids - 1，长度为 len_ids。
#     # test_size 参数表示测试集的比例或样本数量。常用的取值范围是0到1，表示测试集占总体的比例；也可以是一个整数，表示测试集的样本数量。
#     # random_state 参数用于设置随机种子，以确保每次运行时得到相同的随机划分结果。设置相同的随机种子会使得每次运行时的划分结果一致。
#
#     test_indices, val_indices = train_test_split(
#         test_indices, test_size=valid_size,  # random_state=42
#     )
#
#     return train_indices, test_indices, val_indices
#
# a, b, c = split_ids(612)
# print(c)
# print(b)
# print(a)
# import os
# from shutil import copy
# import random
#
# def mkfile(file):
#     if not os.path.exists(file):
#         os.makedirs(file)
#
#
# # 获取data文件夹下所有文件夹名（即需要分类的类名）
# file_path = r'E:\yanghe\TransFuse\TransFuse-main\data\CVC-ClinicDB'
#
# mkfile('dataset/CVC_ClinicDB/test/images')
# mkfile('dataset/CVC_ClinicDB/test/masks')
#
# mkfile('dataset/CVC_ClinicDB/train/images')
# mkfile('dataset/CVC_ClinicDB/train/masks')
# # # 创建 验证集val 文件夹
# mkfile('dataset/CVC_ClinicDB/val/images')
# mkfile('dataset/CVC_ClinicDB/val/masks')
#
# # 划分比例，训练集 : 验证集 = 9 : 1
# # split_rate = 0.5
#
# # 遍历所有类别的全部图像并按比例分成训练集和验证集
#
# cla_path = file_path + '/' +'Original' + '/'  # 某一类别的子目录
# images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
# cla_path_mask = file_path + '/' +'Ground Truth' + '/'  # 某一类别的子目录
# masks = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
# num = len(images)
# # eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
# for index, image in enumerate(images):
#     # eval_index 中保存验证集val的图像名称
#     if np.isin(index, a):
#         image_path = cla_path + image
#         new_path = 'dataset/CVC_ClinicDB/train/images'
#         copy(image_path, new_path)  # 将选中的图像复制到新路径
#
#     # 其余的图像保存在训练集train中
#     elif np.isin(index, b):
#         image_path = cla_path + image
#         new_path = 'dataset/CVC_ClinicDB/test/images'
#         copy(image_path, new_path)
#
#     else:
#         image_path = cla_path + image
#         new_path = 'dataset/CVC_ClinicDB/val/images'
#         copy(image_path, new_path)
#
# for index, image in enumerate(masks):
#     # eval_index 中保存验证集val的图像名称
#     if np.isin(index, a):
#         image_path = cla_path_mask + image
#         new_path = 'dataset/CVC_ClinicDB/train/masks'
#         copy(image_path, new_path)  # 将选中的图像复制到新路径
#
#     # 其余的图像保存在训练集train中
#     elif np.isin(index, b):
#         image_path = cla_path_mask + image
#         new_path = 'dataset/CVC_ClinicDB/test/masks'
#         copy(image_path, new_path)
#
#     else:
#         image_path = cla_path_mask + image
#         new_path = 'dataset/CVC_ClinicDB/val/masks'
#         copy(image_path, new_path)
#
#
# print("processing done!")
#
#
