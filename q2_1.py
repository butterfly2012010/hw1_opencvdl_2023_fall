import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.title = 'Image Smoothing'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(10, 10, 400, 300)

        # Load Image Button
        self.loadImageButton = QPushButton('Load Image 1', self)
        self.loadImageButton.move(10, 10)
        self.loadImageButton.clicked.connect(self.loadImage)

        # Image Label
        self.label = QLabel(self)
        self.label.setGeometry(10, 50, 370, 240)

        # Gaussian Blur Button
        self.gaussianButton = QPushButton('2.1 Gaussian Blur', self)
        self.gaussianButton.move(200, 10)
        self.gaussianButton.clicked.connect(self.gaussianBlur)

        self.show()

    def loadImage(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        pixmap = QPixmap(self.fileName)
        self.label.setPixmap(pixmap)
        self.label.adjustSize()

    def gaussianBlur(self):
        def onChange(value):
            m = cv2.getTrackbarPos('Radius', 'Gaussian Blur')
            blurred = cv2.GaussianBlur(image, (2*m+1, 2*m+1), 0)
            cv2.imshow('Gaussian Blur', blurred)

        image = cv2.imread(self.fileName)
        cv2.imshow('Gaussian Blur', image)
        cv2.createTrackbar('Radius', 'Gaussian Blur', 1, 5, onChange)

app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())
