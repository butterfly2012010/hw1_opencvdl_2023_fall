# import numpy as np
import os

def main():
    # A = np.array([[1, 10], [100, 1000]])
    # B = np.array([[1, 2], [3, 4]])

    # print(np.inner(A, B))
    # print(np.dot(A, B))
    # # print(np.matmul(A, B))
    # print("Hello World!"
    #       "This is a test.")

    # print(os.path.basename("E:/YN/opencvdl/Hw1/Dataset_OpenCvDl_Hw1/Q5_image/Q5_1/cat.jpg"))

    # use opencv, cv2.Sobel() to implement Sobel operator
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load image and convert to grayscale
    img = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q3_image/building.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use api cv2.Sobel()
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate gradient angles in degrees
    angle = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 360
    # Create masks for the given angle ranges
    mask1 = ((angle >= 120) & (angle <= 180)).astype(np.uint8) * 255
    mask2 = ((angle >= 210) & (angle <= 330)).astype(np.uint8) * 255
    # Apply masks to the gradient magnitude using cv2.bitwise_and
    result1 = cv2.bitwise_and(gray, gray, mask=mask1)
    result2 = cv2.bitwise_and(gray, gray, mask=mask2)
    # Display the results
    cv2.imshow('Sobel X', grad_x)
    cv2.imshow('Sobel Y', grad_y)
    combined_window = np.hstack((result1, result2))
    cv2.imshow('Result1 and Result2', combined_window)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # use opencv, cv2.filter2D() to implement Sobel operator
    # import cv2
    # import numpy as np
    # import matplotlib.pyplot as plt

    # # Load image and convert to grayscale
    # img = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q3_image/building.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # Gaussian smoothing
    # smoothed = cv2.GaussianBlur(gray, (3, 3), 1)
    # # Define the Sobel x and y operators
    # sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # # Apply custom filter for Sobel operations
    # grad_x = cv2.filter2D(smoothed, -1, sobel_x)
    # grad_y = cv2.filter2D(smoothed, -1, sobel_y)
    # # Combine Sobel x and Sobel y
    # magnitude = np.sqrt(grad_x**2 + grad_y**2).round().astype('uint8')
    # # Normalize combination result to 0~255
    # normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # # Set threshold
    # threshold_value = 128
    # _, thresholded = cv2.threshold(normalized_magnitude, threshold_value, 255, cv2.THRESH_BINARY)
    # # Calculate gradient angles in degrees
    # angle = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 360
    # # Create masks for the given angle ranges
    # mask1 = ((angle >= 120) & (angle <= 180)).astype(np.uint8) * 255
    # mask2 = ((angle >= 210) & (angle <= 330)).astype(np.uint8) * 255
    # # Apply masks to the gradient magnitude using cv2.bitwise_and
    # result1 = cv2.bitwise_and(normalized_magnitude, normalized_magnitude, mask=mask1)
    # result2 = cv2.bitwise_and(normalized_magnitude, normalized_magnitude, mask=mask2)
    # # Display the results
    # cv2.imshow('Sobel X', grad_x)
    # cv2.imshow('Sobel Y', grad_y)
    # combined_window = np.hstack((normalized_magnitude, thresholded))
    # cv2.imshow('Combined and Thresholded Sobel', combined_window)
    # combined_window = np.hstack((result1, result2))
    # cv2.imshow('Result1 and Result2', combined_window)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()