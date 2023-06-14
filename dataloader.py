import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms

classes = {"airplane": 0, "automobile": 1}

train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class GetData(Dataset):
    # 初始化为整个class提供全局变量，为后续方法提供一些量
    def __init__(self, root_dir, label_dir, transform):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path_list = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]  # 只获取了文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 每个图片的位置
        # 读取图片
        img = Image.open(img_item_path).convert("RGB")
        img = self.transform(img)
        label = classes[self.label_dir]
        return img, label

    def __len__(self):
        return len(self.img_path_list)


root = "./train"
airplane_dir = "airplane"
automobile_dir = "automobile"
airplane_dataset = GetData(root, airplane_dir, train_transform)
automobile_dataset = GetData(root, automobile_dir, train_transform)
train_dataset = airplane_dataset + automobile_dataset

root = "./test"
airplane_dir = "airplane"
automobile_dir = "automobile"
airplane_dataset = GetData(root, airplane_dir, train_transform)
automobile_dataset = GetData(root, automobile_dir, train_transform)
test_dataset = airplane_dataset + automobile_dataset

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

# for i, data in enumerate(train_loader, 0):
#     # 获取输入
#     inputs, labels = data
#     print(inputs.shape)
#     print(labels.shape)
#     break
