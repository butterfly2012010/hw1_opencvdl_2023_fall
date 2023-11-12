import sys
import os
# from ast import main
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import cv2
from matplotlib import scale
import numpy as np

from PIL import Image
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


# from PyQt5 import QtWidgets, QtGui
# import sys
app = QtWidgets.QApplication(sys.argv)

Form = QtWidgets.QWidget()
Form.setWindowTitle('oxxo.studio')
Form.resize(300, 300)

grview = QtWidgets.QGraphicsView(Form)  # 加入 QGraphicsView
grview.setGeometry(20, 20, 260, 200)    # 設定 QGraphicsView 位置與大小
scene = QtWidgets.QGraphicsScene()      # 加入 QGraphicsScene
scene.setSceneRect(0, 0, 300, 400)      # 設定 QGraphicsScene 位置與大小
img = QtGui.QPixmap('E:/YN/opencvdl/Hw1/Dataset_OpenCvDl_Hw1/Q5_image/Q5_4/airplane.png')         # 加入圖片
scene.addPixmap(img)                    # 將圖片加入 scene
grview.setScene(scene)                  # 設定 QGraphicsView 的場景為 scene

Form.show()
sys.exit(app.exec_())