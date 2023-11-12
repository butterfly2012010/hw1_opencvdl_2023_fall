import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.title = 'Image Smoothing with Median Filter'
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

        # Median Filter Button
        self.medianButton = QPushButton('2.3 Median Filter', self)
        self.medianButton.move(200, 10)
        self.medianButton.clicked.connect(self.medianFilter)

        self.show()

    def loadImage(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        pixmap = QPixmap(self.fileName)
        self.label.setPixmap(pixmap)
        self.label.adjustSize()

    def medianFilter(self):
        def onChange(value):
            m = cv2.getTrackbarPos('Radius', 'Median Filter')
            filtered = cv2.medianBlur(image, ksize=2*m+1)
            cv2.imshow('Median Filter', filtered)

        image = cv2.imread(self.fileName)
        cv2.imshow('Median Filter', image)
        cv2.createTrackbar('Radius', 'Median Filter', 1, 5, onChange)

app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())
