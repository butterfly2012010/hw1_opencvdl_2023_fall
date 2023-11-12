import cv2
import numpy as np

# 讀取圖片
image = cv2.imread("./Hw1/Dataset_OpenCvDl_Hw1/Q4_image/burger.png")

# 設置旋轉中心、角度和縮放尺度
center = (240, 200)  # 漢堡在原始圖片中的中心
angle = 30  # 旋轉30度
scale = 0.9  # 縮放尺度為0.9

# 計算旋轉矩陣
M = cv2.getRotationMatrix2D(center, angle, scale)

# 設置平移量
tx = 775 - 240
ty = 535 - 200

# 更新旋轉矩陣以包含平移
M[0, 2] += tx
M[1, 2] += ty

# 使用cv.warpAffine()進行旋轉、縮放和平移
transformed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# 顯示和保存結果
cv2.imshow('Transformed Burger', transformed)
cv2.imwrite("transformed_burger.png", transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
