#!/usr/bin/env python
# -*- coding:utf-8 -*-

import copy
from os import path
import joblib
from time import  *
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from math import ceil
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
from  torchvision.transforms import InterpolationMode
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.models as models
from flask import Flask, request, jsonify
import requests
import base64
import os
import torch.nn.functional as F

from densenet import DenseNet
from feat_tools import *
from scipy.optimize.linesearch import LineSearchWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transformations = transforms.Compose(
    [
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.RandomAffine(degrees=0, shear=(-0.05, 0.05)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((128, 128), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ]
)

test_transformations = transforms.Compose(
    [
        transforms.Resize([128, 128], 0),
        transforms.ToTensor(),
    ]
)


def get_alexnet():
    model = models.alexnet(pretrained=True)
    print('alexnet model is', model)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)
    return model


app = Flask(__name__)


@app.route('/')
def welcome():
    return "Welcome"


@app.route('/prediction', methods=['GET'])
def prediction():
    class_name = ["Non Corona", "Corona"]
    multi_class_name = ["Mild", "Moderate", "Severe"]
    # 模型调用
    model = torch.load("model_res18.pt")
    model_s = torch.load("model_res18_256.pt")
    # 参数调用
    # model = get_alexnet()
    # model.load_state_dict(torch.load("para.pt"))
    model.to(device)
    model.eval()
    model_s.to(device)
    model_s.eval()

    imgpath = "testImg/test/pic.png"
    # imgpath = imgpath
    img = Image.open(imgpath).convert("RGB")
    img = test_transformations(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    pred = model(img)
    output = torch.squeeze(pred)
    predict = torch.softmax(output, dim=0)
    index = torch.argmax(predict).numpy()
    index_s = 0
    print('predicted:', class_name[index], predict[index].item())
    if index == 1:
        pred_s = model_s(img)
        output_s = torch.squeeze(pred_s)
        predict_s = torch.softmax(output_s, dim=0)
        index_s = torch.argmax(predict_s).numpy()
        class_name[index] + "," + str(predict[index].item()) + "," + str(multi_class_name[index_s])
    os.remove(imgpath)
    return class_name[index]+","+str(predict[index].item())+","+str(multi_class_name[index_s])


@app.route('/DR_prediction', methods=['GET'])
def DR_prediction():
    class_name = ["Healthy", "Corona", "Viral Pneumonia"]
    # 模型调用
    model = torch.load("DR_model_res18.pt")
    # 参数调用
    # model = get_alexnet()
    # model.load_state_dict(torch.load("para.pt"))
    model.to(device)
    model.eval()

    imgpath = "testImg/test/pic.png"
    img = Image.open(imgpath).convert("RGB")
    img = test_transformations(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    pred = model(img)
    output = torch.squeeze(pred)
    predict = torch.softmax(output, dim=0)
    index = torch.argmax(predict).numpy()
    print('predicted:', class_name[index], predict[index].item())
    os.remove(imgpath)
    # 直接将成的图片删除了
    return class_name[index]+","+str(predict[index].item())


@app.route('/savePic', methods=['POST'])
def save():
    file = request.form.get('file')
    print('file is', file)
    print('the type of file is', type(file))
    temp = str.split(file, ',')
    print('temp is', temp)
    url = temp[1]
    print('url is ', url + '=' * (4 - len(url) % 4))
    img = base64.urlsafe_b64decode(url + '=' * (4 - len(url) % 4))
    if url is None:
        return "No File"
    else:
        filepath = "./testImg/test/pic.png"
        img_data = np.frombuffer(img, np.uint8)
        print(type(img_data))
        img_arr = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
        print(type(img_arr))
        print(img_arr.shape)
        print(np.max(img_arr))
        print(np.min(img_arr))
        cv2.imwrite('testImg/gray.png', img_arr)
        print('---' * 100)

        img_arr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        print(type(img_arr))
        print(img_arr.shape)
        print(np.max(img_arr))
        print(np.min(img_arr))
        cv2.imwrite('testImg/color1.png', img_arr)
        # print('shapes:', img.shape)
        with open(filepath, 'wb') as fd:
            fd.write(img)
        return "Load Successfully"


@app.route('/predict_breast_cancer_degree', methods=['POST'])
def predict_breast_cancer_degree():
    file = request.form.get('file')
    # start_x = float(request.form.get('start_x'))
    # start_y = float(request.form.get('start_y'))
    # end_x = float(request.form.get('end_x'))
    # end_y = float(request.form.get('end_y'))
    # original_x = float(request.form.get('original_x'))
    # original_y = float(request.form.get('original_y'))
    # params = dict(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y,
    #               original_x=original_x, original_y=original_y)
    # print(params)
    # 首先需要获取ROI
    data_url = str.split(file, ',')[1]
    print(type(data_url))
    img_data = base64.urlsafe_b64decode(data_url + '=' * (4 - len(data_url) % 4))
    # print('original img data type:', type(img_data))
    img_data = np.frombuffer(img_data, np.uint8)
    # print(type(img_data))
    img_arr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    # print(type(img_arr))
    # img_arr = np.asarray(img_arr)
    # print('img arr shape:', img_arr.shape)
    # print('max:', np.max(img_arr))
    # print('min:', np.min(img_arr))
    # cv2.imwrite('testImg/degree.png', img_arr)

    height, width, depth = img_arr.shape
    # original_height, original_width = original_y, original_x
    # y_scale_factor = height / original_height
    # x_scale_factor = width / original_width
    # params = dict(height=height, width=width, depth=depth, original_width=original_width, original_height=original_height,
    #               x_scale_factor=x_scale_factor, y_scale_factor=y_scale_factor)
    # print(params)
    #
    # scaled_start_x = int(x_scale_factor * start_x)
    # scaled_start_y = int(y_scale_factor * start_y)
    # scaled_end_x = int(x_scale_factor * end_x)
    # scaled_end_y = int(y_scale_factor * end_y)
    # params = dict(scaled_start_x=scaled_start_x, scaled_start_y=scaled_start_y,
    #               scaled_end_x=scaled_end_x, scaled_end_y=scaled_end_y)
    # print(params)
    # # 依据ROI来获取对应的Mask
    # mask = img_arr[scaled_start_y:scaled_end_y + 1, scaled_start_x:scaled_end_x + 1, :]
    # print('mask shape:', mask.shape)
    # cv2.imwrite('testImg/mask.png', mask)

    hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
    yellow_low_hsv = np.array([26, 43, 46])
    yellow_high_hsv = np.array([34, 255, 255])
    yellow_mask = cv2.inRange(hsv, lowerb=yellow_low_hsv, upperb=yellow_high_hsv)
    yellow_contours, yellow_hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,  # EXTERNAL选择最外框
                                                         cv2.CHAIN_APPROX_SIMPLE)
    contours = yellow_contours
    max_width = -1
    max_height = -1
    max_contours = None
    max_box = None
    max_rect = None
    for i in range(len(contours)):
        # contours[0] = np.concatenate((contours[0], contours[i]), axis=0)
        rect = cv2.minAreaRect(contours[i])  # 找到最小外接矩形，该矩形可能有方向
        cur_width = rect[1][1]
        cur_height = rect[1][0]
        box = cv2.boxPoints(rect)
        if cur_width >= max_width and cur_height >= max_height:
            max_width = cur_width
            max_height = cur_height
            max_contours = contours[i]
            max_box = box
            max_rect = rect
        # print(i, ":", rect)
        # print(box)

    # print('max rect:', max_rect)
    # print('max box:', max_box)
    # print('max contours:', max_contours)
    left_point_x = np.min(max_box[:, 0])
    right_point_x = np.max(max_box[:, 0])
    top_point_y = np.min(max_box[:, 1])
    bottom_point_y = np.max(max_box[:, 1])
    delta_h = 15
    delta_w = 15
    crop_img = img_arr[int(top_point_y) + delta_h:int(bottom_point_y + 1) - delta_h,
           int(left_point_x) + delta_w:int(right_point_x + 1) - delta_w, :]

    img = cv2.resize(crop_img, (256, 256))
    # cv2.imwrite('testImg/mask3.png', resized_img)
    mask = get_topo_mask(img)
    # print('mask shape:', mask.shape)
    # print('mask dtype:', mask.dtype)
    # cv2.imwrite('testImg/mask4.png', mask)
    # print('img shape:', img.dtype)
    # 对ROI和Mask进行resize

    # 送入库函数来获取形状和纹理特征
    texture_feats = generate_single_texture_features(img, mask)
    print('--'*10 + 'text feats' + '--'*10)
    print(texture_feats)
    print(len(texture_feats))

    # 送入拓扑特征提取函数获取拓扑特征
    mask = mask.astype(np.uint8)
    # print('true mask', mask)
    topo_feats = generate_single_topo_features(mask)
    print('--'*10 + 'topo feats' + '--'*10)
    print(topo_feats)
    print(len(topo_feats))

    # 合并两类特征
    if len(topo_feats) != 0:
        merged_feats = dict(texture_feats, **topo_feats)
    else:
        merged_feats = texture_feats
    print('--'*10 + 'merged feats' + '--'*10)
    print(len(merged_feats))

    cancer_degree = -1
    try:
        # 根据之前筛选出来的特征关键词来获取有效特征
        feats_csv_file_1ist = [r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features.csv',
                               r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features_with_cv.csv',
                               r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features_2.csv',
                               r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features_3.csv',
                               ]
        file_idx = 0
        filtered_feats = []

        df = pd.read_csv(feats_csv_file_1ist[file_idx])
        df_columns = df.columns.tolist()[3:]
        for col in df_columns:
            filtered_feats.append(merged_feats[col])
        filtered_feats = np.asarray(filtered_feats)
        filtered_feats = filtered_feats.reshape((1, len(filtered_feats)))

        # 加载模型参数文件
        # class_weight = {0: 0.3, 1: 0.7}
        # svc_model = SVC(
        #     kernel='rbf',
        #     class_weight=class_weight,
        #     # probability=True,
        #     probability=False,
        #     # gamma=float(1 / 20),
        #     random_state=1,
        # )
        model = joblib.load('./sklearn_model/svc_model.pkl')
        cancer_degree = model.predict(filtered_feats)
    except Exception as e:
       print('the exception is:', e)
    print('cancer degree:', cancer_degree[0])
    return str(cancer_degree[0])
    # 预测良恶性


@app.route('/predict_breast_cancer_degree_2', methods=['POST'])
def predict_breast_cancer_degree_2():
    file = request.form.get('file')
    data_url = str.split(file, ',')[1]
    print(type(data_url))

    img_data = base64.urlsafe_b64decode(data_url + '=' * (4 - len(data_url) % 4))
    img_data = np.frombuffer(img_data, np.uint8)
    img_arr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    cv2.imwrite("./uploadImages/" + str(time()) + "_img.png", img_arr)
    height, width, depth = img_arr.shape
    hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
    yellow_low_hsv = np.array([26, 43, 46])
    yellow_high_hsv = np.array([34, 255, 255])
    yellow_mask = cv2.inRange(hsv, lowerb=yellow_low_hsv, upperb=yellow_high_hsv)
    yellow_contours, yellow_hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,  # EXTERNAL选择最外框
                                                         cv2.CHAIN_APPROX_SIMPLE)
    contours = yellow_contours
    max_width = -1
    max_height = -1
    max_contours = None
    max_box = None
    max_rect = None
    print('这里面可能找不到最大值和最小值')
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])  # 找到最小外接矩形，该矩形可能有方向
        cur_width = rect[1][1]
        cur_height = rect[1][0]
        box = cv2.boxPoints(rect)
        if cur_width >= max_width and cur_height >= max_height:
            max_width = cur_width
            max_height = cur_height
            max_contours = contours[i]
            max_box = box
            max_rect = rect

    left_point_x = np.min(max_box[:, 0])
    right_point_x = np.max(max_box[:, 0])
    top_point_y = np.min(max_box[:, 1])
    bottom_point_y = np.max(max_box[:, 1])
    delta_h = 15
    delta_w = 15
    crop_img = img_arr[int(top_point_y) + delta_h:int(bottom_point_y + 1) - delta_h,
           int(left_point_x) + delta_w:int(right_point_x + 1) - delta_w, :]
    cv2.imwrite("./uploadImages/" + str(time()) + '_cropped_img.png', crop_img)
    img = cv2.resize(crop_img, (256, 256))
    mask = get_topo_mask(img)
    cv2.imwrite("./uploadImages/" + str(time()) + "_mask.png", mask)

    # 送入库函数来获取形状和纹理特征
    texture_feats = generate_single_texture_features(img, mask)
    print('--'*10 + 'text feats' + '--'*10)
    # print(texture_feats)
    print(len(texture_feats))

    cancer_degree = -1
    cancer_degree_prob = -1
    try:
        topo_feats_keys = ['Degree', 'Vertex', 'Subgraph', 'Component', 'Average', 'Points', 'Diameter']
        # 根据之前筛选出来的特征关键词来获取有效特征
        feats_csv_file_1ist = [r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features.csv',
                               r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features_with_cv.csv',
                               r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features_2.csv',
                               r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features_3.csv',
                               ]
        file_idx = 0
        filtered_feats = []

        df = pd.read_csv(feats_csv_file_1ist[file_idx])
        df_columns = df.columns.tolist()[3:]

        scale_range_set = []
        for filtered_feat in df_columns:
            for key in topo_feats_keys:
                if key in filtered_feat:
                    scale = filtered_feat.split('_')[-1]
                    scale_range_set.append(scale)
                    break
        scale_range_set = set(scale_range_set)
        print('scale_range_set:', scale_range_set)
        scale_range_set = [int(i) for i in scale_range_set]

        # 送入拓扑特征提取函数获取拓扑特征
        mask = mask.astype(np.uint8)
        topo_feats = generate_single_topo_features(mask, scale_range_set)
        print('--' * 10 + 'topo feats' + '--' * 10)
        print(topo_feats)
        print(len(topo_feats))

        # # 合并两类特征
        if len(topo_feats) != 0:
            merged_feats = dict(texture_feats, **topo_feats)
        else:
            merged_feats = texture_feats
        print('--' * 10 + 'merged feats' + '--' * 10)
        print(len(merged_feats))
        # print(merged_feats)

        for col in df_columns:
            filtered_feats.append(merged_feats[col])
        print('final feats:', filtered_feats)
        print(len(filtered_feats))
        filtered_feats = np.asarray(filtered_feats)

        print('filtered feats num:', len(filtered_feats))
        filtered_feats = filtered_feats.reshape((1, len(filtered_feats)))
        filtered_feats = (filtered_feats - np.min(filtered_feats)) / (np.max(filtered_feats) - np.min(filtered_feats) + 1e-7)

        model = joblib.load('./sklearn_model/svc_model2.pkl')
        cancer_degree_probility_arr = model.predict_proba(filtered_feats)[0]
        print('cancer_degree_probility_arr', cancer_degree_probility_arr)
        cancer_degree = np.argmax(cancer_degree_probility_arr)
        print('cancer degree:', cancer_degree)
        cancer_degree_prob = cancer_degree_probility_arr[cancer_degree]
        print('cancer degree_prob:', cancer_degree_prob)
    except Exception as e:
       print('the exception is:', e)
    result = {'degree': str(cancer_degree), 'prob': str(cancer_degree_prob)}
    return jsonify(result)
    # 预测良恶性



@app.route('/predict_breast_cancer_degree_3', methods=['POST'])
def predict_breast_cancer_degree_3():

    # 获取传送过来的完整原始图像
    file = request.form.get('file')
    print('file type', type(file))
    print('file is', file)
    data_url = str.split(file, ',')[1]
    print('data_url type:', type(data_url))
    print('data_url:', data_url)

    img_data = base64.urlsafe_b64decode(data_url + '=' * (4 - len(data_url) % 4))
    print('type data 1:', type(img_data))
    img_data = np.frombuffer(img_data, np.uint8)
    print('type data 2:', type(img_data))
    print(img_data.shape)
    img_arr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    print('type data 3:', type(img_data))
    print(img_data.shape)
    cv2.imwrite("./uploadImages/" + str(time()) + "_img.png", img_arr)

    # 提取感兴趣区ROI
    height, width, depth = img_arr.shape
    hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
    yellow_low_hsv = np.array([26, 43, 46])
    yellow_high_hsv = np.array([34, 255, 255])
    yellow_mask = cv2.inRange(hsv, lowerb=yellow_low_hsv, upperb=yellow_high_hsv)
    yellow_contours, yellow_hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,  # EXTERNAL选择最外框
                                                         cv2.CHAIN_APPROX_SIMPLE)
    contours = yellow_contours
    max_width = -1
    max_height = -1
    max_contours = None
    max_box = None
    max_rect = None
    print('这里面可能找不到最大值和最小值')
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])  # 找到最小外接矩形，该矩形可能有方向
        cur_width = rect[1][1]
        cur_height = rect[1][0]
        box = cv2.boxPoints(rect)
        if cur_width >= max_width and cur_height >= max_height:
            max_width = cur_width
            max_height = cur_height
            max_contours = contours[i]
            max_box = box
            max_rect = rect

    left_point_x = np.min(max_box[:, 0])
    right_point_x = np.max(max_box[:, 0])
    top_point_y = np.min(max_box[:, 1])
    bottom_point_y = np.max(max_box[:, 1])
    delta_h = 15
    delta_w = 15
    # 矩形勾画工具在矩形的四个边角处都画上了小圆圈，这对提取ROI造成了一定的干扰
    crop_img = img_arr[int(top_point_y) + delta_h:int(bottom_point_y + 1) - delta_h,
           int(left_point_x) + delta_w:int(right_point_x + 1) - delta_w, :]
    cv2.imwrite("./uploadImages/" + str(time()) + '_cropped_img.png', crop_img)
    img = cv2.resize(crop_img, (256, 256))
    mask = get_topo_mask(img)
    cv2.imwrite("./uploadImages/" + str(time()) + "_mask.png", mask)

    has_calcification = judge_mask(mask)

    if not has_calcification:
        return {'degree': str(-1), 'prob': str(-1),
                'no_calcification_point': True}

    # 送入库函数来获取形状和纹理特征
    texture_feats = generate_single_texture_features(img, mask)
    print('--'*10 + 'text feats' + '--'*10)
    # print(texture_feats)
    print(len(texture_feats))

    cancer_degree = -1
    cancer_degree_prob = -1
    no_calcification_point = False
    try:
        topo_feats_keys = ['Degree', 'Vertex', 'Subgraph', 'Component', 'Average', 'Points', 'Diameter']
        # 根据之前筛选出来的特征关键词来获取有效特征
        feats_csv_file_1ist = [r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features.csv',
                               r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features_with_cv.csv',
                               r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features_2.csv',
                               r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features_3.csv',
                               ]
        file_idx = 0
        filtered_feats = []

        df = pd.read_csv(feats_csv_file_1ist[file_idx])
        df_columns = df.columns.tolist()[3:]

        scale_range_set = []
        for filtered_feat in df_columns:
            for key in topo_feats_keys:
                if key in filtered_feat:
                    scale = filtered_feat.split('_')[-1]
                    scale_range_set.append(scale)
                    break
        scale_range_set = set(scale_range_set)
        print('scale_range_set:', scale_range_set)
        scale_range_set = [int(i) for i in scale_range_set]

        # 送入拓扑特征提取函数获取拓扑特征
        mask = mask.astype(np.uint8)
        topo_feats = generate_single_topo_features(mask, scale_range_set)
        print('--' * 10 + 'topo feats' + '--' * 10)
        # print(topo_feats)
        print(len(topo_feats))

        # # 合并两类特征
        if len(topo_feats) != 0:
            merged_feats = dict(texture_feats, **topo_feats)
        else:
            merged_feats = texture_feats
        print('--' * 10 + 'merged feats' + '--' * 10)
        print(len(merged_feats))
        # print(merged_feats)

        for col in df_columns:
            filtered_feats.append(merged_feats[col])
        # print('final feats:', filtered_feats)
        print(len(filtered_feats))
        filtered_feats = np.asarray(filtered_feats)

        """需要再加上使用深度学习模型跑出来的特征 deep_feats
        """
        # 输入：获取原始图片，将其转化为torch模型所需要的形式
        # 模型：运行模型，得到deep_feats张量
        # 输出：将张量转化为numpy形式
        # 与原始特征进行合并
        print(crop_img.shape)
        crop_img_transform = cv2.resize(crop_img, (256, 256))
        print(crop_img_transform.shape)
        crop_img_transform = crop_img_transform.reshape(1, 3, crop_img_transform.shape[0], crop_img_transform.shape[1])
        print('transformed shape:', crop_img_transform.shape)
        crop_img_transform_tensor = torch.Tensor(crop_img_transform)
        model = DenseNet()
        model.load_state_dict(torch.load('deep_model/manual_densenet_epoch_89_0.7907_best.pkl',
                                         map_location='cpu'))
        feats = model.features(crop_img_transform_tensor)
        feats = F.relu(feats, inplace=True)
        feats = F.avg_pool2d(feats, kernel_size=feats.size()[-1], stride=1).view(feats.size()[0], -1)
        feats = model.classifier1(feats)
        feats_arr = feats.detach().numpy()
        feats_arr = feats_arr.reshape(-1)
        print('deep model feats shape:', feats_arr.shape)
        print(filtered_feats.shape)
        filtered_feats = np.concatenate([feats_arr, filtered_feats])
        print('final filtered feats shape:', filtered_feats.shape)

        print('filtered feats num:', len(filtered_feats))
        filtered_feats = filtered_feats.reshape((1, len(filtered_feats)))
        filtered_feats = (filtered_feats - np.min(filtered_feats)) / (np.max(filtered_feats) - np.min(filtered_feats) + 1e-7)

        model = joblib.load('./sklearn_model/svc_model3.pkl')
        cancer_degree_probility_arr = model.predict_proba(filtered_feats)[0]
        print('cancer_degree_probility_arr', cancer_degree_probility_arr)
        cancer_degree = np.argmax(cancer_degree_probility_arr)
        print('cancer degree:', cancer_degree)
        cancer_degree_prob = cancer_degree_probility_arr[cancer_degree]
        print('cancer degree_prob:', cancer_degree_prob)
    except Exception as e:
       print('the exception is:', e)
    result = {'degree': str(cancer_degree), 'prob': str(cancer_degree_prob),
              'no_calcification_point': False}
    return jsonify(result)
    # 预测良恶性


@app.route('/predict_breast_cancer_degree_4', methods=['POST'])
def predict_breast_cancer_degree_4():

    # 获取传送过来的完整原始图像
    file = request.form.get('file')
    print('file type', type(file))
    print('file is', file)
    data_url = str.split(file, ',')[1]
    print('data_url type:', type(data_url))
    print('data_url:', data_url)

    img_data = base64.urlsafe_b64decode(data_url + '=' * (4 - len(data_url) % 4))
    print('type data 1:', type(img_data))
    img_data = np.frombuffer(img_data, np.uint8)
    print('type data 2:', type(img_data))
    print(img_data.shape)
    img_arr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    print('type data 3:', type(img_data))
    print(img_data.shape)
    cv2.imwrite("./uploadImages/" + str(time()) + "_img.png", img_arr)

    # 提取感兴趣区ROI
    height, width, depth = img_arr.shape
    hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
    yellow_low_hsv = np.array([26, 43, 46])
    yellow_high_hsv = np.array([34, 255, 255])
    yellow_mask = cv2.inRange(hsv, lowerb=yellow_low_hsv, upperb=yellow_high_hsv)
    yellow_contours, yellow_hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,  # EXTERNAL选择最外框
                                                         cv2.CHAIN_APPROX_SIMPLE)
    contours = yellow_contours
    max_width = -1
    max_height = -1
    max_contours = None
    max_box = None
    max_rect = None
    print('这里面可能找不到最大值和最小值')
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])  # 找到最小外接矩形，该矩形可能有方向
        cur_width = rect[1][1]
        cur_height = rect[1][0]
        box = cv2.boxPoints(rect)
        if cur_width >= max_width and cur_height >= max_height:
            max_width = cur_width
            max_height = cur_height
            max_contours = contours[i]
            max_box = box
            max_rect = rect

    left_point_x = np.min(max_box[:, 0])
    right_point_x = np.max(max_box[:, 0])
    top_point_y = np.min(max_box[:, 1])
    bottom_point_y = np.max(max_box[:, 1])
    delta_h = 15
    delta_w = 15
    # 矩形勾画工具在矩形的四个边角处都画上了小圆圈，这对提取ROI造成了一定的干扰
    crop_img = img_arr[int(top_point_y) + delta_h:int(bottom_point_y + 1) - delta_h,
           int(left_point_x) + delta_w:int(right_point_x + 1) - delta_w, :]
    cv2.imwrite("./uploadImages/" + str(time()) + '_cropped_img.png', crop_img)
    img = cv2.resize(crop_img, (256, 256))
    mask = get_topo_mask(img)
    cv2.imwrite("./uploadImages/" + str(time()) + "_mask.png", mask)

    has_calcification = judge_mask(mask)

    if not has_calcification:
        return {'degree': str(-1), 'prob': str(-1),
                'no_calcification_point': True}

    cancer_degree = -1
    cancer_degree_prob = -1

    crop_img_transform = cv2.resize(crop_img, (256, 256))
    print(crop_img_transform.shape)
    crop_img_transform = crop_img_transform.reshape(1, 3, crop_img_transform.shape[0], crop_img_transform.shape[1])
    print('transformed shape:', crop_img_transform.shape)
    crop_img_transform_tensor = torch.Tensor(crop_img_transform)

    model = DenseNet()
    model.load_state_dict(torch.load('deep_model/manual_densenet_epoch_89_0.7907_best.pkl',
                                     map_location='cpu'))
    predict_probs = model.forward(crop_img_transform_tensor)
    predict_probs_normalized = F.softmax(predict_probs)
    predict_probs_normalized_arr = predict_probs_normalized.detach().numpy()
    predict_probs_normalized_arr = predict_probs_normalized_arr.reshape(-1)

    print(predict_probs_normalized_arr)
    cancer_degree = np.argmax(predict_probs_normalized_arr)
    cancer_degree_prob = predict_probs_normalized_arr[cancer_degree]
    result = {'degree': str(cancer_degree), 'prob': str(cancer_degree_prob),
              'no_calcification_point': False}
    print('res:', result)
    return jsonify(result)


def test():
    test_path = "testImg/test/Covid"
    paths = glob.glob(path.join(test_path, "*.png"))
    for Imgpath in paths:
        print(Imgpath)
        prediction(Imgpath)


# def main():
#     # "data/covid_data/"  "testImg/val_im/" "testImg/test/"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)
