# 最后使用手动生成tensor数据，并没有读取数据，但是二者的效果是相同的
import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# img1 = plt.imread('cat.jfif')
# print(img1.shape)
# print(img1)
# plt.imshow(img1)
# plt.show()


normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformer_train = transforms.Compose([
        # transforms.Resize(300),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ])

train_dataset = datasets.ImageFolder(
        root = '../image/data/',
        transform = transformer_train,
    )

train_loader = dataloader.DataLoader(
        train_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 1,
        pin_memory = True,
    )
for i in enumerate(train_loader):
    print(i)