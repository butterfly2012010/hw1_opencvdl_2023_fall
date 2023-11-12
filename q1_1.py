import cv2
import numpy as np


# 讀取圖片
image = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q1_image/rgb.jpg')

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
