#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import copy
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
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

# 数据预处理

data = pd.read_csv("CT_data_label.csv")
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

test_transformations = transforms.Compose(
    [
        transforms.Resize([256, 256], 0),
        transforms.ToTensor(),
    ]
)


# %%

def plot_classes(Y_train, Y_dev):
    # (non corona = 0, corona = 1)
    train_uniques, train_uniques_count = np.unique(Y_train, return_counts=True)
    dev_uniques, dev_uniques_count = np.unique(Y_dev, return_counts=True)

    train_uniques = train_uniques.astype(np.object)
    train_uniques[train_uniques == 0] = "Non Corona"
    train_uniques[train_uniques == 1] = "Corona"

    dev_uniques = dev_uniques.astype(np.object)
    dev_uniques[dev_uniques == 0] = "Non Corona"
    dev_uniques[dev_uniques == 1] = "Corona"

    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    bar1 = plt.bar(train_uniques, train_uniques_count)
    bar1[0].set_color("g")
    bar1[1].set_color("r")
    plt.xlabel("Category")
    plt.ylabel("No of pictures")
    plt.title("Train set")
    plt.subplot(122)
    bar2 = plt.bar(dev_uniques, dev_uniques_count)
    bar2[0].set_color("g")
    bar2[1].set_color("r")
    plt.xlabel("Category")
    plt.ylabel("No of pictures")
    plt.title("Dev test")
    plt.show()


# %%


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


# %%
def get_transformations(for_train=True, resize=(256, 256)):
    transformations = {
        "train_transforms": transforms.Compose(
            [
                transforms.RandomRotation(degrees=(-5, 5)),  # 随机旋转
                transforms.RandomAffine(degrees=0, shear=(-0.05, 0.05)),  # 随机转化灰度图
                transforms.RandomHorizontalFlip(0.5),  # 随机上下翻转
                transforms.Resize(resize),
                transforms.ToTensor(),
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


# %%

def show_images(img_path, transforms=None):
    global img1
    img = Image.open(img_path).convert("RGB")
    if transforms:
        img1 = transforms(img)
        img1 = img1.permute(2, 1, 0)

    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Normal")
    if transforms:
        plt.subplot(122)
        plt.imshow(img1)
        plt.title("Transformed")
    plt.show()


# %%

# show_images(img_path="data/covid_data/Corona2_9.jpg", transforms=None)

# %%

# show_images(img_path="data/covid_data/Corona2_9.jpg", transforms=get_transformations(for_train=True))

# %%

BATCH_SIZE = 8

# %%


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#  %%
def get_ResModel(flag):
    model = models.resnet18(pretrained=flag)
    features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=features, out_features=2)
    model = model.to(device)
    return model


# %%
def get_vgg16(flag):
    model = models.vgg16(pretrained=flag)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)
    model = model.to(device)
    return model


# %%

def get_alexnet(flag):
    model = models.alexnet(pretrained=flag)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)
    model = model.to(device)
    return model


# %%

def train_step(model, inputs, labels, criterion, optimizer):
    optimizer.zero_grad()

    preds = model(inputs)
    loss = criterion(preds, labels)

    loss.backward()
    optimizer.step()

    return preds, loss


# %%

def eval_step(model, inputs, labels, criterion):
    preds = model(inputs)
    loss = criterion(preds, labels)

    return preds, loss


# %%

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


# %%

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


# %%

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
        X, Y, train_size=0.8, random_state=19, shuffle=True
    )
    print("No of train samples: {}".format(len(X_train)))
    print("No of dev samples: {}".format(len(X_dev)))
    # 展示训练集和测试集图片类别及数量
    plot_classes(Y_train, Y_dev)
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
    return train(model, train_dataset, dev_dataset, criterion, optimizer, 5)


# %%
def prediction(model, imgpath):
    class_name = ["Non Corona", "Corona"]
    img = Image.open(imgpath).convert("RGB")
    img = test_transformations(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    res = model(img)
    output = torch.squeeze(res)
    predict = torch.softmax(output, dim=0)
    p = predict.tolist()[1]
    predict_cla = torch.argmax(predict).numpy()
    print('predicted:', class_name[predict_cla], predict[predict_cla].item())
    return class_name[predict_cla], p


def test(n):
    X_train, X_dev, Y_train, Y_dev = train_test_split(
        X, Y, train_size=0.8, shuffle=True
    )
    model = KNeighborsClassifier(n)
    k = KFold(n_splits=n)
    score = cross_val_score(model, X_train, Y_train, cv=k)
    return score


def main():
    model = model_training()
    torch.save(model, "model_res18.pt")
    '''''''''
    model_alex = torch.load("model_newAlex.pt")
    # model_vgg = torch.load("model_newVGG16.pt")
    paths = glob.glob("testImg/test/Covid/*")
    paths_1 = glob.glob("testImg/test/Normal/*")
    paths_2 = glob.glob("testImg/test/Viral Pneumonia/*")
    counter = len(paths) + len(paths_1) + len(paths_2)
    print("dev Corona: ", len(paths), "dev Non Corona: ", len(paths_1) + len(paths_2))
    cor = 0
    y_preds = []
    y_labels = []
    yps = []
    yls = []
    wrong_list = ["Wrong picture: "]
    for Imgpath in paths:
        print(Imgpath)
        pred1, p = prediction(model_alex, Imgpath)
        y_labels.append(1)
        y_preds.append(p)
        pred, p = prediction(model_vgg, Imgpath)
        yls.append(1)
        yps.append(p)
        if pred == "Corona":
            cor = cor + 1
        else:
            wrong_list.append(Imgpath)
            
    for Imgpath in paths_1:
        print(Imgpath)
        pred1, p = prediction(model_alex, Imgpath)
        y_labels.append(0)
        y_preds.append(p)
        pred, p = prediction(model_vgg, Imgpath)
        yls.append(0)
        yps.append(p)
        if pred == "Non Corona":
            cor = cor + 1
        else:
            wrong_list.append(Imgpath)

    for Imgpath in paths_2:
        print(Imgpath)
        pred1, p = prediction(model_alex, Imgpath)
        y_labels.append(0)
        y_preds.append(p)
        pred, p = prediction(model_vgg, Imgpath)
        yls.append(0)
        yps.append(p)
        if pred == "Non Corona":
            cor = cor + 1
        else:
            wrong_list.append(Imgpath)
            
    acc = cor / counter
    print("dev_acc: ", acc)
    y_label = np.asarray(y_labels)
    y_pred = np.asarray(y_preds)
    y_l = np.asarray(yls)
    y_p = np.asarray(yps)
    fpr, tpr, thresholds_keras = roc_curve(y_label, y_pred, pos_label=1)
    f, t, th = roc_curve(y_l, y_p, pos_label=1)
    Auc = auc(fpr, tpr)
    Auc_v = auc(f, t)
    print("AUC : ", Auc)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC_alexNet = {:.4f}'.format(Auc))
    plt.plot(f, t, color='r', label='AUC_vgg16Net = {:.4f}'.format(Auc_v))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    for wrong_path in wrong_list:
        print(wrong_path)
'''''''''


if __name__ == '__main__':
    main()
