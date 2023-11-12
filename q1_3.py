import cv2

# 讀取圖片
image = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q1_image/rgb.jpg')

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
