#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import copy
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.models as models
import glob

data = pd.read_csv("DR_data_label.csv")
trans1 = transforms.RandomRotation(degrees=(-10, 10))
trans2 = transforms.RandomCrop(256, 256)
trans3 = transforms.RandomVerticalFlip(1)
index = 0
labellist = []
imglist = []
for path in data.x:
    imgs = Image.open(path).convert("RGB")
    label = data.y[index]
    index = index + 1
    img1 = trans1(imgs)
    img2 = trans2(imgs)
    img3 = trans3(imgs)
    temp = [imgs, img1, img2, img3]
    for timg in temp:
        timg = np.array(timg)
        imglist.append(timg)
        labellist.append(label)

X, Y = imglist, np.asarray(labellist)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
# base_path = "slices/"
test_transformations = transforms.Compose(
    [
        transforms.Resize([128, 128], 0),
        transforms.ToTensor()
    ]
)


class CovidDataset(object):
    def __init__(self, X, Y, transforms=None):
        self.X = X
        self.Y = Y
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = Image.fromarray(self.X[idx])
        # img_path = self.X[idx]
        img_label = self.Y[idx]

        # img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        # print(img_path)
        # print(img.shape)
        # print(img_label)
        return img, img_label


def get_transformations(for_train=True, resize=(256, 256)):
    transformations = {
        "train_transforms": transforms.Compose(
            [
                transforms.RandomRotation(degrees=(-5, 5)),  # 随机旋转
                transforms.RandomAffine(degrees=0, shear=(-0.05, 0.05)),  # 随机转化灰度图
                transforms.RandomHorizontalFlip(0.5),  # 随机上下翻转
                transforms.Resize(resize),
                transforms.ToTensor()
            ]
        ),
        "test_transforms": transforms.Compose(
            [transforms.Resize(resize), transforms.ToTensor()]
        ),
    }
    if for_train:
        return transformations["train_transforms"]
    else:
        return transformations["test_transforms"]


def get_ResModel(flag):
    model = models.resnet18(pretrained=flag)
    features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=features, out_features=3)
    model = model.to(device)
    return model


def get_vcg16(flag):
    model = models.vgg16(pretrained=flag)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=3)
    model = model.to(device)
    return model


def get_alexnet(flag):
    model = models.alexnet(pretrained=flag)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=3)
    model = model.to(device)
    return model


def train_step(model, inputs, labels, criterion, optimizer):
    optimizer.zero_grad()

    preds = model(inputs)
    loss = criterion(preds, labels)

    loss.backward()
    optimizer.step()

    return preds, loss


def eval_step(model, inputs, labels, criterion):
    preds = model(inputs)
    loss = criterion(preds, labels)

    return preds, loss


def train_epoch(model, train_dataset, criterion, optimizer, batch_size):
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    correct_count = 0
    total_loss = 0

    model.train()
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds, loss = train_step(model, imgs, labels, criterion, optimizer)

        preds = torch.argmax(preds, dim=1)
        correct_count += (preds == labels).sum().item()
        total_loss += loss.item()

    return correct_count / len(train_dataset), total_loss / len(train_dataset)


def eval_epoch(model, dev_dataset, criterion, batch_size):
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)

    correct_count = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for imgs, labels in dev_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds, loss = eval_step(model, imgs, labels, criterion)

            preds = torch.argmax(preds, dim=1)
            correct_count += (preds == labels).sum().item()
            total_loss = loss.item()

    return correct_count / len(dev_dataset), total_loss / len(dev_dataset)


def train(model, train_dataset, dev_dataset, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        train_acc, train_loss = train_epoch(
            model, train_dataset, criterion, optimizer, BATCH_SIZE
        )
        print("train_acc: {:.4f}, train_loss: {:.4f}".format(train_acc, train_loss))

        dev_acc, dev_loss = eval_epoch(model, dev_dataset, criterion, BATCH_SIZE)
        print("dev_acc: {:.4f}, dev_loss: {:.4f}".format(dev_acc, dev_loss))

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


def model_training():
    # 数据集划分
    X_train, X_dev, Y_train, Y_dev = train_test_split(
        X, Y, train_size=0.8, random_state=12, shuffle=True
    )
    print("No of train samples: {}".format(len(X_train)))
    print("No of dev samples: {}".format(len(X_dev)))
    # 模型载入
    flag = True
    model = get_ResModel(flag)
    # 数据预处理
    train_dataset = CovidDataset(X_train, Y_train, get_transformations(for_train=True))
    dev_dataset = CovidDataset(X_dev, Y_dev, get_transformations(for_train=False))
    # 损失函数（交叉熵损失函数）
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # 优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # 模型训练
    model = train(model, train_dataset, dev_dataset, criterion, optimizer, 5)
    # 混淆矩阵
    dev_loader = DataLoader(dataset=dev_dataset, shuffle=True)
    y_true = []
    y_pred = []
    for imgs, labels in dev_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        preds = model(imgs)
        pred = torch.argmax(preds, dim=1)
        print(pred, labels)
        y_pred.append(pred)
        y_true.append(labels)
    print(confusion_matrix(y_true, y_pred))
    return model


def prediction(model, imgpath):
    classname = ["Normal", "Corona", "Viral Pneumonia"]
    # classname = ["Mild", "Moderate", "Severe"]
    img = Image.open(imgpath).convert("RGB")
    img = test_transformations(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    # print(imgpath)
    # print('predicted:', classname[predict_cla], predict[predict_cla].item())
    return classname[predict_cla]


def main():
    model_trained = torch.load("DR_model_res18.pt")
    paths_0 = glob.glob("Covid19-dataset/test/Covid/*")
    paths_1 = glob.glob("Covid19-dataset/test/Normal/*")
    paths_2 = glob.glob("Covid19-dataset/test/Viral Pneumonia/*")
    paths = [paths_1, paths_0, paths_2]
    index = 0
    counter = 0
    CN = ["Normal", "Corona", "Viral Pneumonia"]
    print("test Corona: ", len(paths_0), "test Non Corona: ", len(paths_1), "test Viral Pneumonia: ", len(paths_2))
    total = len(paths_0)+len(paths_1)+len(paths_2)
    for p in paths:
        for imgpath in p:
            pred = prediction(model_trained, imgpath)
            if CN[index] == pred:
                counter = counter+1
        index = index+1

    print("correct: ", counter, "accuracy: ", counter/total)
    if counter/total > 0.8:
        torch.save(model_trained, "DR_model_res18.pt")



if __name__ == '__main__':
    main()
