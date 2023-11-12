import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.title = 'Image Smoothing with Bilateral Filter'
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

        # Bilateral Filter Button
        self.bilateralButton = QPushButton('2.2 Bilateral Filter', self)
        self.bilateralButton.move(200, 10)
        self.bilateralButton.clicked.connect(self.bilateralFilter)

        self.show()

    def loadImage(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        pixmap = QPixmap(self.fileName)
        self.label.setPixmap(pixmap)
        self.label.adjustSize()

    def bilateralFilter(self):
        def onChange(value):
            m = cv2.getTrackbarPos('Radius', 'Bilateral Filter')
            filtered = cv2.bilateralFilter(image, d=2*m+1, sigmaColor=90, sigmaSpace=90)
            cv2.imshow('Bilateral Filter', filtered)

        image = cv2.imread(self.fileName)
        cv2.imshow('Bilateral Filter', image)
        cv2.createTrackbar('Radius', 'Bilateral Filter', 1, 5, onChange)

app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())
