import numpy as np
import random
import multiprocessing

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data

from .dataset import SegDataset


def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )
    # np.linspace(0, len_ids - 1, len_ids).astype("int")，表示输入的数组。
    # 数组使用 np.linspace(0, len_ids - 1, len_ids) 生成一个整数类型的等间隔数组，范围从 0 到 len_ids - 1，长度为 len_ids。
    # test_size 参数表示测试集的比例或样本数量。常用的取值范围是0到1，表示测试集占总体的比例；也可以是一个整数，表示测试集的样本数量。
    # random_state 参数用于设置随机种子，以确保每次运行时得到相同的随机划分结果。设置相同的随机种子会使得每次运行时的划分结果一致。

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices


def get_dataloaders(input_paths, target_paths, batch_size, if_train=True):

    transform_input4train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((352, 352)), transforms.Grayscale()]
    )

    if if_train:
        train_dataset = SegDataset(
            input_paths=input_paths,
            target_paths=target_paths,
            transform_input=transform_input4train,
            transform_target=transform_target,
            hflip=True,
            vflip=True,
            affine=True,
        )
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4  # multiprocessing.Pool()._processes,
        )
        return train_dataloader
    else:
    # test_dataset = SegDataset(
    #     input_paths=input_paths,
    #     target_paths=target_paths,
    #     transform_input=transform_input4test,
    #     transform_target=transform_target,
    # )
        val_dataset = SegDataset(
            input_paths=input_paths,
            target_paths=target_paths,
            transform_input=transform_input4test,
            transform_target=transform_target,
        )
        val_dataloader = data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4  # multiprocessing.Pool()._processes,
        )
        return val_dataloader

    # train_indices, test_indices, val_indices = split_ids(len(input_paths))
    #
    # train_dataset = data.Subset(train_dataset, train_indices)
    # val_dataset = data.Subset(val_dataset, val_indices)
    # test_dataset = data.Subset(test_dataset, test_indices)

    # train_dataloader = data.DataLoader(
    #     dataset=train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=4  # multiprocessing.Pool()._processes,
    # )

    # test_dataloader = data.DataLoader(
    #     dataset=test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=4  # multiprocessing.Pool()._processes,
    # )

    # val_dataloader = data.DataLoader(
    #     dataset=val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=4  # multiprocessing.Pool()._processes,
    # )

    # return train_dataloader, test_dataloader, val_dataloader



