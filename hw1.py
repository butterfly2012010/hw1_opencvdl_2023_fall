# -*- coding: utf-8 -*-


import sys
import os
# from ast import main
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QWidget, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import cv2
from matplotlib import scale
import numpy as np

from PIL import Image, ImageQt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation  #, RandomCrop, ColorJitter, RandomRotation
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt


class Ui_hw1(object):
    def __init__(self, hw1) -> None:
        self.setupUi(hw1)
        self.retranslateUi(hw1)

        # click event
        # click LoadImage1
        self.LoadImage1.clicked.connect(self.loadImage1)
        # click LoadImage2
        self.LoadImage2.clicked.connect(self.loadImage2)

        # 1. Image Processing
        # 1.1 Color Seperation
        self.ColorSeperation.clicked.connect(self.q1_1)
        # 1.2 Color Transformation
        self.ColorTransformation.clicked.connect(self.q1_2)
        # 1.3 Color Extraction
        self.ColorExtraction.clicked.connect(self.q1_3)

        # 2. Image Smoothing
        # 2.1 Gaussian Blur
        self.GaussianBlur.clicked.connect(self.q2_1)
        # 2.2 Bilateral Filter
        self.BilateralFilter.clicked.connect(self.q2_2)
        # 2.3 Median Filter
        self.MedianFilter.clicked.connect(self.q2_3)

        # 3. Edge Detection
        # 3.1 Sobel X
        self.SobelX.clicked.connect(self.q3_1)
        # 3.2 Sobel Y
        self.SobelY.clicked.connect(self.q3_2)
        # 3.3 Combination and Threshold
        self.CombinationAndThreshold.clicked.connect(self.q3_3)
        # 3.4 Gradient Angle
        self.GradientAngle.clicked.connect(self.q3_4)

        # 4. Transforms
        self.Transforms_2.clicked.connect(self.q4)

        # 5. VGG19
        # 5.1 Load Image
        self.LoadImage.clicked.connect(self.openImage)
        # 5.1 Show Augmented Images
        self.ShowAugmentedImages.clicked.connect(self.q5_1)
        # 5.2 Show Model Structure
        self.ShowModelStructure.clicked.connect(self.q5_2)
        # 5.3 Show Acc and Loss
        self.ShowAccAndLoss.clicked.connect(self.q5_3)
        # 5.4 Inference
        self.Inference.clicked.connect(self.q5_4)

    def setupUi(self, hw1):
        hw1.setObjectName("hw1")
        hw1.resize(662, 720)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(hw1.sizePolicy().hasHeightForWidth())
        hw1.setSizePolicy(sizePolicy)
        self.ImageProcessing = QtWidgets.QGroupBox(hw1)
        self.ImageProcessing.setGeometry(QtCore.QRect(140, 40, 201, 151))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageProcessing.sizePolicy().hasHeightForWidth())
        self.ImageProcessing.setSizePolicy(sizePolicy)
        self.ImageProcessing.setObjectName("ImageProcessing")
        self.ColorSeperation = QtWidgets.QPushButton(self.ImageProcessing)
        self.ColorSeperation.setGeometry(QtCore.QRect(30, 20, 131, 23))
        self.ColorSeperation.setObjectName("ColorSeperation")
        self.ColorTransformation = QtWidgets.QPushButton(self.ImageProcessing)
        self.ColorTransformation.setGeometry(QtCore.QRect(30, 60, 131, 23))
        self.ColorTransformation.setObjectName("ColorTransformation")
        self.ColorExtraction = QtWidgets.QPushButton(self.ImageProcessing)
        self.ColorExtraction.setGeometry(QtCore.QRect(30, 100, 131, 23))
        self.ColorExtraction.setObjectName("ColorExtraction")
        self.ImageSmoothing = QtWidgets.QGroupBox(hw1)
        self.ImageSmoothing.setGeometry(QtCore.QRect(140, 220, 201, 181))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageSmoothing.sizePolicy().hasHeightForWidth())
        self.ImageSmoothing.setSizePolicy(sizePolicy)
        self.ImageSmoothing.setObjectName("ImageSmoothing")
        self.GaussianBlur = QtWidgets.QPushButton(self.ImageSmoothing)
        self.GaussianBlur.setGeometry(QtCore.QRect(30, 30, 131, 23))
        self.GaussianBlur.setObjectName("GaussianBlur")
        self.MedianFilter = QtWidgets.QPushButton(self.ImageSmoothing)
        self.MedianFilter.setGeometry(QtCore.QRect(30, 110, 131, 23))
        self.MedianFilter.setObjectName("MedianFilter")
        self.BilateralFilter = QtWidgets.QPushButton(self.ImageSmoothing)
        self.BilateralFilter.setGeometry(QtCore.QRect(30, 70, 131, 23))
        self.BilateralFilter.setObjectName("BilateralFilter")
        self.EdgeDetection = QtWidgets.QGroupBox(hw1)
        self.EdgeDetection.setGeometry(QtCore.QRect(140, 430, 201, 231))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EdgeDetection.sizePolicy().hasHeightForWidth())
        self.EdgeDetection.setSizePolicy(sizePolicy)
        self.EdgeDetection.setObjectName("EdgeDetection")
        self.SobelX = QtWidgets.QPushButton(self.EdgeDetection)
        self.SobelX.setGeometry(QtCore.QRect(30, 30, 131, 23))
        self.SobelX.setObjectName("SobelX")
        self.CombinationAndThreshold = QtWidgets.QPushButton(self.EdgeDetection)
        self.CombinationAndThreshold.setGeometry(QtCore.QRect(30, 110, 131, 31))
        self.CombinationAndThreshold.setObjectName("CombinationAndThreshold")
        self.SobelY = QtWidgets.QPushButton(self.EdgeDetection)
        self.SobelY.setGeometry(QtCore.QRect(30, 70, 131, 23))
        self.SobelY.setObjectName("SobelY")
        self.GradientAngle = QtWidgets.QPushButton(self.EdgeDetection)
        self.GradientAngle.setGeometry(QtCore.QRect(30, 150, 131, 23))
        self.GradientAngle.setObjectName("GradientAngle")
        self.Transforms = QtWidgets.QGroupBox(hw1)
        self.Transforms.setGeometry(QtCore.QRect(390, 40, 191, 201))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Transforms.sizePolicy().hasHeightForWidth())
        self.Transforms.setSizePolicy(sizePolicy)
        self.Transforms.setObjectName("Transforms")
        self.Rotation = QtWidgets.QLabel(self.Transforms)
        self.Rotation.setGeometry(QtCore.QRect(10, 30, 51, 16))
        self.Rotation.setObjectName("Rotation")
        self.Scaling = QtWidgets.QLabel(self.Transforms)
        self.Scaling.setGeometry(QtCore.QRect(10, 60, 51, 16))
        self.Scaling.setObjectName("Scaling")
        self.Tx = QtWidgets.QLabel(self.Transforms)
        self.Tx.setGeometry(QtCore.QRect(10, 90, 51, 16))
        self.Tx.setObjectName("Tx")
        self.Ty = QtWidgets.QLabel(self.Transforms)
        self.Ty.setGeometry(QtCore.QRect(10, 120, 51, 16))
        self.Ty.setObjectName("Ty")
        self.Transforms_2 = QtWidgets.QPushButton(self.Transforms)
        self.Transforms_2.setGeometry(QtCore.QRect(30, 160, 131, 23))
        self.Transforms_2.setObjectName("Transforms_2")
        self.lineEdit_Rotation = QtWidgets.QLineEdit(self.Transforms)
        self.lineEdit_Rotation.setGeometry(QtCore.QRect(60, 30, 71, 20))
        self.lineEdit_Rotation.setObjectName("lineEdit_Rotation")
        self.lineEdit_Scaling = QtWidgets.QLineEdit(self.Transforms)
        self.lineEdit_Scaling.setGeometry(QtCore.QRect(60, 60, 71, 20))
        self.lineEdit_Scaling.setObjectName("lineEdit_Scaling")
        self.lineEdit_Tx = QtWidgets.QLineEdit(self.Transforms)
        self.lineEdit_Tx.setGeometry(QtCore.QRect(60, 90, 71, 20))
        self.lineEdit_Tx.setObjectName("lineEdit_Tx")
        self.lineEdit_Ty = QtWidgets.QLineEdit(self.Transforms)
        self.lineEdit_Ty.setGeometry(QtCore.QRect(60, 120, 71, 20))
        self.lineEdit_Ty.setObjectName("lineEdit_Ty")
        self.deg = QtWidgets.QLabel(self.Transforms)
        self.deg.setGeometry(QtCore.QRect(140, 30, 31, 16))
        self.deg.setObjectName("deg")
        self.pixel_1 = QtWidgets.QLabel(self.Transforms)
        self.pixel_1.setGeometry(QtCore.QRect(140, 90, 31, 16))
        self.pixel_1.setObjectName("pixel_1")
        self.pixel_2 = QtWidgets.QLabel(self.Transforms)
        self.pixel_2.setGeometry(QtCore.QRect(140, 120, 31, 16))
        self.pixel_2.setObjectName("pixel_2")
        self.VGG19 = QtWidgets.QGroupBox(hw1)
        self.VGG19.setGeometry(QtCore.QRect(390, 260, 191, 401))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.VGG19.sizePolicy().hasHeightForWidth())
        self.VGG19.setSizePolicy(sizePolicy)
        self.VGG19.setObjectName("VGG19")
        self.LoadImage = QtWidgets.QPushButton(self.VGG19)
        self.LoadImage.setGeometry(QtCore.QRect(30, 20, 131, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LoadImage.sizePolicy().hasHeightForWidth())
        self.LoadImage.setSizePolicy(sizePolicy)
        self.LoadImage.setObjectName("LoadImage")
        self.ShowAugmentedImages = QtWidgets.QPushButton(self.VGG19)
        self.ShowAugmentedImages.setGeometry(QtCore.QRect(30, 50, 131, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShowAugmentedImages.sizePolicy().hasHeightForWidth())
        self.ShowAugmentedImages.setSizePolicy(sizePolicy)
        self.ShowAugmentedImages.setObjectName("ShowAugmentedImages")
        self.ShowModelStructure = QtWidgets.QPushButton(self.VGG19)
        self.ShowModelStructure.setGeometry(QtCore.QRect(30, 90, 131, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShowModelStructure.sizePolicy().hasHeightForWidth())
        self.ShowModelStructure.setSizePolicy(sizePolicy)
        self.ShowModelStructure.setObjectName("ShowModelStructure")
        self.ShowAccAndLoss = QtWidgets.QPushButton(self.VGG19)
        self.ShowAccAndLoss.setGeometry(QtCore.QRect(30, 120, 131, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShowAccAndLoss.sizePolicy().hasHeightForWidth())
        self.ShowAccAndLoss.setSizePolicy(sizePolicy)
        self.ShowAccAndLoss.setObjectName("ShowAccAndLoss")
        self.Inference = QtWidgets.QPushButton(self.VGG19)
        self.Inference.setGeometry(QtCore.QRect(30, 150, 131, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Inference.sizePolicy().hasHeightForWidth())
        self.Inference.setSizePolicy(sizePolicy)
        self.Inference.setObjectName("Inference")
        self.label_Predict = QtWidgets.QLabel(self.VGG19)
        self.label_Predict.setGeometry(QtCore.QRect(10, 190, 51, 16))
        self.label_Predict.setObjectName("label_Predict")
        self.graphicsView_PredictImage = QtWidgets.QGraphicsView(self.VGG19)
        self.graphicsView_PredictImage.setEnabled(True)
        self.graphicsView_PredictImage.setGeometry(QtCore.QRect(10, 220, 171, 171))
        self.graphicsView_PredictImage.setToolTip("")
        self.graphicsView_PredictImage.setStatusTip("")
        self.graphicsView_PredictImage.setWhatsThis("")
        self.graphicsView_PredictImage.setAccessibleName("")
        self.graphicsView_PredictImage.setAccessibleDescription("")
        self.graphicsView_PredictImage.setInputMethodHints(QtCore.Qt.ImhNone)
        self.graphicsView_PredictImage.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.graphicsView_PredictImage.setObjectName("graphicsView_PredictImage")

        self.scene = QGraphicsScene()
        self.graphicsView_PredictImage.setScene(self.scene)

        self.LoadImage1 = QtWidgets.QPushButton(hw1)
        self.LoadImage1.setGeometry(QtCore.QRect(40, 270, 75, 23))
        self.LoadImage1.setObjectName("LoadImage1")
        self.LoadImage2 = QtWidgets.QPushButton(hw1)
        self.LoadImage2.setGeometry(QtCore.QRect(40, 340, 75, 23))
        self.LoadImage2.setObjectName("LoadImage2")
        self.FileName_Image1 = QtWidgets.QLabel(hw1)
        self.FileName_Image1.setGeometry(QtCore.QRect(40, 300, 47, 12))
        self.FileName_Image1.setText("")
        self.FileName_Image1.setObjectName("FileName_Image1")
        self.FileName_Image2 = QtWidgets.QLabel(hw1)
        self.FileName_Image2.setGeometry(QtCore.QRect(40, 370, 47, 12))
        self.FileName_Image2.setText("")
        self.FileName_Image2.setObjectName("FileName_Image2")
        self.ImageProcessing.raise_()
        self.Transforms.raise_()
        self.LoadImage1.raise_()
        self.EdgeDetection.raise_()
        self.ImageSmoothing.raise_()
        self.LoadImage2.raise_()
        self.VGG19.raise_()
        self.FileName_Image1.raise_()
        self.FileName_Image2.raise_()

        self.retranslateUi(hw1)
        QtCore.QMetaObject.connectSlotsByName(hw1)

    def retranslateUi(self, hw1):
        _translate = QtCore.QCoreApplication.translate
        hw1.setWindowTitle(_translate("hw1", "HW1"))
        self.ImageProcessing.setTitle(_translate("hw1", "1. Image Processing"))
        self.ColorSeperation.setText(_translate("hw1", "1.1 Color Seperation"))
        self.ColorTransformation.setText(_translate("hw1", "1.2 Color Transformation"))
        self.ColorExtraction.setText(_translate("hw1", "1.3 Color Extraction"))
        self.ImageSmoothing.setTitle(_translate("hw1", "2. Image Smoothing"))
        self.GaussianBlur.setText(_translate("hw1", "2.1 Gaussian Blur"))
        self.MedianFilter.setText(_translate("hw1", "2.3 Median Filter"))
        self.BilateralFilter.setText(_translate("hw1", "2.2 Bilateral Filter"))
        self.EdgeDetection.setTitle(_translate("hw1", "3. Edge Detection"))
        self.SobelX.setText(_translate("hw1", "3.1 Sobel X"))
        self.CombinationAndThreshold.setText(_translate("hw1", "3.3 Combination and \n"
"Threshold"))
        self.SobelY.setText(_translate("hw1", "3.2 Sobel Y"))
        self.GradientAngle.setText(_translate("hw1", "3.4 Gradient Angle"))
        self.Transforms.setTitle(_translate("hw1", "4. Transforms"))
        self.Rotation.setText(_translate("hw1", "Rotation: "))
        self.Scaling.setText(_translate("hw1", "Scaling: "))
        self.Tx.setText(_translate("hw1", "Tx: "))
        self.Ty.setText(_translate("hw1", "Ty: "))
        self.Transforms_2.setText(_translate("hw1", "4. Transforms"))
        self.deg.setText(_translate("hw1", "deg"))
        self.pixel_1.setText(_translate("hw1", "pixel"))
        self.pixel_2.setText(_translate("hw1", "pixel"))
        self.VGG19.setTitle(_translate("hw1", "5. VGG19"))
        self.LoadImage.setText(_translate("hw1", "Load Image"))
        self.ShowAugmentedImages.setText(_translate("hw1", "5.1 Show Augmented \n"
"Images"))
        self.ShowModelStructure.setText(_translate("hw1", "5.2 Show Model Structure"))
        self.ShowAccAndLoss.setText(_translate("hw1", "5.3 Show Acc and Loss"))
        self.Inference.setText(_translate("hw1", "5.4 Inference"))
        self.label_Predict.setText(_translate("hw1", "Predict = "))
        self.LoadImage1.setText(_translate("hw1", "Load Image1"))
        self.LoadImage2.setText(_translate("hw1", "Load Image2"))

    def loadImage1(self):
        options = QtWidgets.QFileDialog.Options()
        self.fileName1, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        self.FileName_Image1.setText(os.path.basename(self.fileName1))

    def loadImage2(self):
        options = QtWidgets.QFileDialog.Options()
        self.fileName2, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        self.FileName_Image2.setText(os.path.basename(self.fileName2))

    def q1_1(self):
        # import cv2
        # import numpy as np


        # 讀取圖片
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # image = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q1_image/rgb.jpg')
        image = cv2.imread(fileName)

        # 使用 cv2.split() 分離圖片的RGB通道
        B, G, R = cv2.split(image)

        # 使用 cv2.merge() 將單一通道的灰階圖像轉回BGR圖像
        B_img = cv2.merge([B, np.zeros_like(B), np.zeros_like(B)])
        G_img = cv2.merge([np.zeros_like(G), G, np.zeros_like(G)])
        R_img = cv2.merge([np.zeros_like(R), np.zeros_like(R), R])

        # 顯示每個通道的圖片
        cv2.imshow('Blue Channel', B_img)
        cv2.imshow('Green Channel', G_img)
        cv2.imshow('Red Channel', R_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q1_2(self):
        # import cv2
        # import numpy as np

        # 讀取圖片
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # image = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q1_image/rgb.jpg')
        image = cv2.imread(fileName)

        # (i) 使用 cv2.cvtColor() 轉換圖像為灰度圖像
        I_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # (ii) 使用 cv2.split() 分離圖片的BGR通道
        B, G, R = cv2.split(image)

        # 計算 I_2
        I_2 = ((R + G + B) / 3).astype(np.uint8)

        # 顯示轉換後的圖片
        cv2.imshow('Grayscale Image using cv2.cvtColor', I_1)
        cv2.imshow('Grayscale Image using Mean of Channels', I_2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q1_3(self):
        # import cv2

        # 讀取圖片
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # image = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q1_image/rgb.jpg')
        image = cv2.imread(fileName)

        # 1. 將圖片從 BGR 轉換為 HSV 格式
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. 提取黃綠色 mask
        lower_bound = (25, 25, 25)
        upper_bound = (90, 255, 255)  # 黃綠色的H範圍大約在25到90之間
        yellow_green_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        # 3. 將 mask 轉換為 BGR 格式
        mask_bgr = cv2.cvtColor(yellow_green_mask, cv2.COLOR_GRAY2BGR)

        # 4. 利用 mask 移除圖片中的黃綠色，生成 I_2
        I_2 = cv2.bitwise_not(mask_bgr, image, mask=yellow_green_mask)

        # 顯示結果
        cv2.imshow('Yellow-Green Mask in BGR', mask_bgr)
        cv2.imshow('Image without Yellow-Green', I_2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q2_1(self):
        def onChange(value):
                m = cv2.getTrackbarPos('Radius', 'Gaussian Blur')
                blurred = cv2.GaussianBlur(image, (2*m+1, 2*m+1), 0)
                cv2.imshow('Gaussian Blur', blurred)

        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # image = cv2.imread("E:/YN/opencvdl/Hw1/Dataset_OpenCvDl_Hw1/Q2_image/image1.jpg")
        image = cv2.imread(fileName)
        cv2.imshow('Gaussian Blur', image)
        cv2.createTrackbar('Radius', 'Gaussian Blur', 1, 5, onChange)

    def q2_2(self):
        def onChange(value):
                m = cv2.getTrackbarPos('Radius', 'Bilateral Filter')
                filtered = cv2.bilateralFilter(image, d=2*m+1, sigmaColor=90, sigmaSpace=90)
                cv2.imshow('Bilateral Filter', filtered)

        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # image = cv2.imread("E:/YN/opencvdl/Hw1/Dataset_OpenCvDl_Hw1/Q2_image/image1.jpg")
        image = cv2.imread(fileName)
        cv2.imshow('Bilateral Filter', image)
        cv2.createTrackbar('Radius', 'Bilateral Filter', 1, 5, onChange)

    def q2_3(self):
        def onChange(value):
            m = cv2.getTrackbarPos('Radius', 'Median Filter')
            filtered = cv2.medianBlur(image, ksize=2*m+1)
            cv2.imshow('Median Filter', filtered)

        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # image = cv2.imread("E:/YN/opencvdl/Hw1/Dataset_OpenCvDl_Hw1/Q2_image/image1.jpg")
        image = cv2.imread(fileName)
        cv2.imshow('Median Filter', image)
        cv2.createTrackbar('Radius', 'Median Filter', 1, 5, onChange)

    def q3_1(self):
        def custom_filter2D(image, kernel):
            """Apply custom 2D filter operation"""
            h, w = image.shape
            kh, kw = kernel.shape
            pad_size = kh // 2
            padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
            output = np.zeros_like(image)

            for i in range(h):
                for j in range(w):
                    region = padded_image[i:i+kh, j:j+kw]
                    output[i, j] = np.sum(region * kernel)
            
            return output

        # Load image and convert to grayscale
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # img = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q3_image/building.jpg')
        img = cv2.imread(fileName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian smoothing
        smoothed = cv2.GaussianBlur(gray, (3, 3), 1)

        # Define the Sobel x and y operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # Apply custom filter for Sobel operations
        grad_x = custom_filter2D(smoothed, sobel_x)

        # Display the results
        cv2.imshow('Sobel X', grad_x)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q3_2(self):
        def custom_filter2D(image, kernel):
            """Apply custom 2D filter operation"""
            h, w = image.shape
            kh, kw = kernel.shape
            pad_size = kh // 2
            padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
            output = np.zeros_like(image)

            for i in range(h):
                for j in range(w):
                    region = padded_image[i:i+kh, j:j+kw]
                    output[i, j] = np.sum(region * kernel)
            
            return output

        # Load image and convert to grayscale
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # img = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q3_image/building.jpg')
        img = cv2.imread(fileName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian smoothing
        smoothed = cv2.GaussianBlur(gray, (3, 3), 1)

        # Define the Sobel x and y operators
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply custom filter for Sobel operations
        grad_y = custom_filter2D(smoothed, sobel_y)

        # Display the results
        cv2.imshow('Sobel Y', grad_y)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q3_3(self):
        def custom_filter2D(image, kernel):
            """Apply custom 2D filter operation"""
            h, w = image.shape
            kh, kw = kernel.shape
            pad_size = kh // 2
            padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
            output = np.zeros_like(image)

            for i in range(h):
                for j in range(w):
                    region = padded_image[i:i+kh, j:j+kw]
                    output[i, j] = np.sum(region * kernel)
            
            return output

        # Load image and convert to grayscale
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # img = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q3_image/building.jpg')
        img = cv2.imread(fileName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian smoothing
        smoothed = cv2.GaussianBlur(gray, (3, 3), 1)

        # Define the Sobel x and y operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # (3-1)
        # Apply custom filter for Sobel operations
        grad_x = custom_filter2D(smoothed, sobel_x)
        grad_y = custom_filter2D(smoothed, sobel_y)

        # (3-2)
        # Combine Sobel x and Sobel y
        magnitude = np.sqrt(grad_x**2 + grad_y**2).round().astype('uint8')
        # magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.uint8)

        # Normalize combination result to 0~255
        normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Set threshold
        threshold_value = 128
        _, thresholded = cv2.threshold(normalized_magnitude, threshold_value, 255, cv2.THRESH_BINARY)

        # Display the results
        combined_window = np.hstack((normalized_magnitude, thresholded))
        cv2.imshow('Combined and Thresholded Sobel', combined_window)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q3_4(self):
        def custom_filter2D(image, kernel):
            """Apply custom 2D filter operation"""
            h, w = image.shape
            kh, kw = kernel.shape
            pad_size = kh // 2
            padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
            output = np.zeros_like(image)

            for i in range(h):
                for j in range(w):
                    region = padded_image[i:i+kh, j:j+kw]
                    output[i, j] = np.sum(region * kernel)
            
            return output

        # Load image and convert to grayscale
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # img = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q3_image/building.jpg')
        img = cv2.imread(fileName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian smoothing
        smoothed = cv2.GaussianBlur(gray, (3, 3), 1)

        # Define the Sobel x and y operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # (3-1)
        # Apply custom filter for Sobel operations
        grad_x = custom_filter2D(smoothed, sobel_x)
        grad_y = custom_filter2D(smoothed, sobel_y)

        # (3-2)
        # Combine Sobel x and Sobel y
        magnitude = np.sqrt(grad_x**2 + grad_y**2).round().astype('uint8')
        # magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.uint8)

        # Normalize combination result to 0~255
        normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Set threshold
        threshold_value = 128
        _, thresholded = cv2.threshold(normalized_magnitude, threshold_value, 255, cv2.THRESH_BINARY)

        # (3-3)
        # Calculate gradient angles in degrees
        angle = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 360

        # Create masks for the given angle ranges
        # mask1 = ((angle >= 120) & (angle <= 180)).astype(np.uint8) * 255
        # mask2 = ((angle >= 210) & (angle <= 330)).astype(np.uint8) * 255
        mask1 = np.where((angle >= 120) & (angle <= 180), 255, 0).astype(np.uint8)
        mask2 = np.where((angle >= 210) & (angle <= 330), 255, 0).astype(np.uint8)

        # # Calculate the magnitude of gradients
        # magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply masks to the gradient magnitude using cv2.bitwise_and
        result1 = cv2.bitwise_and(normalized_magnitude, normalized_magnitude, mask=mask1)
        result2 = cv2.bitwise_and(normalized_magnitude, normalized_magnitude, mask=mask2)

        # Display the results
        # cv2.imshow('Sobel X', grad_x)
        # cv2.imshow('Sobel Y', grad_y)

        # # cv2.imshow('Combination of Sobel X and Sobel Y', normalized_magnitude)
        # # cv2.imshow('Thresholded Sobel', normalized_magnitude)
        # combined_window = np.hstack((normalized_magnitude, thresholded))
        # cv2.imshow('Combined and Thresholded Sobel', combined_window)

        combined_window = np.hstack((result1, result2))
        cv2.imshow('Result1 and Result2', combined_window)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q4(self):
        # import cv2
        # import numpy as np

        # 讀取圖片
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        # image = cv2.imread("./Hw1/Dataset_OpenCvDl_Hw1/Q4_image/burger.png")
        image = cv2.imread(fileName)

        # 設置旋轉中心、角度和縮放尺度
        center = (240, 200)  # 漢堡在原始圖片中的中心
        # angle = 30  # 旋轉30度
        # scale = 0.9  # 縮放尺度為0.9
        angle = float(self.lineEdit_Rotation.text())
        scale = float(self.lineEdit_Scaling.text())

        # 計算旋轉矩陣
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # print(M)

        # 設置平移量
        # tx = 775 - 240
        # ty = 535 - 200
        tx = float(self.lineEdit_Tx.text())
        ty = float(self.lineEdit_Ty.text())

        # 更新旋轉矩陣以包含平移
        M[0, 2] += tx
        M[1, 2] += ty
        # print(M)

        # 使用cv.warpAffine()進行旋轉、縮放和平移
        transformed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # 顯示和保存結果
        cv2.imshow('Transformed Burger', transformed)
        cv2.imwrite("transformed_burger.png", transformed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def q5_1(self):
        # from PIL import Image
        # import matplotlib.pyplot as plt
        # import torchvision.transforms as transforms
        # import os
        # from pathlib import Path

        # Define the directory where images are stored
        # image_dir = Path("E:/YN/opencvdl/Hw1/Dataset_OpenCvDl_Hw1/Q5_image/Q5_1/")
        options = QtWidgets.QFileDialog.Options()
        image_dir = Path(QtWidgets.QFileDialog.getExistingDirectory(None, "Open Image Directory", "", options=options))

        # Check if the directory exists and list all image files
        if image_dir.is_dir():
            image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg'))
        else:
            image_files = []

        # Define the transformations
        transformations = {
            'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=1),
            'RandomVerticalFlip': transforms.RandomVerticalFlip(p=1),
            'RandomRotation': transforms.RandomRotation(30)
        }

        # Load images and apply transformations
        augmented_images = {}
        for image_file in image_files:
            image_name = image_file.name
            img = Image.open(image_file)
            augmented_images[image_name] = {}
            for transform_name, transform in transformations.items():
                augmented_images[image_name][transform_name] = transform(img)

        # augmented_images

        #####################################################################################################


        # # Create a figure with subplots
        # fig, axs = plt.subplots(3, 3, figsize=(6, 6))

        # # Flatten the array of axes for easy iteration
        # axs = axs.flatten()

        # # Loop over the first nine images and transformations and plot
        # for i, image_file in enumerate(image_files[:9]):
        #     img = Image.open(image_file)
        #     axs[i].imshow(img)
        #     axs[i].set_title(f'Original Image {i+1}')
        #     axs[i].axis('off')

        # plt.show()

        # Apply transformations and plot
        # fig, axs = plt.subplots(len(transformations), 3, figsize=(6, 6))
        fig, axs = plt.subplots(3, 3, figsize=(6, 6))

        # for i, transform in enumerate(transformations.values()):
        #     for j, image_file in enumerate(image_files):  # apply each transformation to three images
        #         img = Image.open(image_file)
        #         axs[i, j].imshow(transform(img))
        #         axs[i, j].set_title(f'Augmentation {i+1} on Image {j+1}')
        #         axs[i, j].axis('off')

        # Flatten the array of axes for easy iteration
        axs = axs.flatten()

        for i, image_file in enumerate(image_files):
            img = Image.open(image_file)
            img = transformations['RandomHorizontalFlip'](img)
            img = transformations['RandomVerticalFlip'](img)
            img = transformations['RandomRotation'](img)
            axs[i].imshow(img)
            axs[i].set_title(os.path.split(image_file)[1].split('.')[0])
            axs[i].axis('off')


        plt.show()

    def q5_2(self):
        # Check if CUDA is available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # RTX 2080 SUPER

        # VGG19 with Batch Normalization
        model = models.vgg19_bn(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 10) # Adjusting for CIFAR10's 10 classes
        model = model.to(device)

        # Print the model summary
        summary(model, (3, 32, 32))  # CIFAR10 images have a size of 32x32 with 3 color channels

    def q5_3(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)

        # use PIL to plot the image from fileName
        with Image.open(fileName) as image:
            image.show()

    def openImage(self):
            imagePath, _ = QFileDialog.getOpenFileName()
            if imagePath:
                self.inputImage = cv2.imread(imagePath)
                self.inputImage = cv2.cvtColor(self.inputImage, cv2.COLOR_BGR2RGB)

                self.currentImage = QPixmap(imagePath).scaled(128, 128)
                self.scene.addPixmap(self.currentImage)
                self.graphicsView_PredictImage.fitInView(self.scene.itemsBoundingRect())

    def performInference(self):
        class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']

        input_tensor = self.transform_image(self.inputImage)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(input_tensor.unsqueeze(0))
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

        self.label_Predict.setText(f'Predict = {class_name[predicted.item()]}')

        # Display histogram
        plt.figure(figsize=(10, 8))
        # plt.bar(range(len(probabilities)), probabilities.numpy())
        plt.bar(class_name, probabilities.numpy())
        plt.xlabel('Classes')
        plt.ylabel('Probability (%)')
        plt.title('Probability Distribution of Model Predictions')
        plt.xticks(rotation=45)
        plt.show()

    def transform_image(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(image)

    def load_model(self, model_path):
        model = models.vgg19_bn(pretrained=False)  # `pretrained=False` is deprecated
        model.classifier[6] = nn.Linear(4096, 10) # Adjusting for CIFAR10's 10 classes
        model.load_state_dict(torch.load(model_path))
        return model

    def q5_4(self):
        print(f'Load model weights from model_path...')
        self.model_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Model", "", "All Files (*);;pth (*.pth)")
        # model = self.load_model('path_to_your_model.pth')
        self.model = self.load_model(self.model_path)
        print(f'Model loaded successfully!')
        self.performInference()


            

def main():
    app = QtWidgets.QApplication(sys.argv)
    hw1 = QtWidgets.QWidget()
    ui = Ui_hw1(hw1)
    # ui.setupUi(hw1)
    hw1.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()