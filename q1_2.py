import cv2
import numpy as np

# 讀取圖片
image = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q1_image/rgb.jpg')

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
