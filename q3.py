import cv2
import numpy as np

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
img = cv2.imread('./Hw1/Dataset_OpenCvDl_Hw1/Q3_image/building.jpg')
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
mask1 = np.where((angle >= 120) & (angle <= 180), 255, 0).astype(np.uint8)
mask2 = np.where((angle >= 210) & (angle <= 330), 255, 0).astype(np.uint8)

# # Calculate the magnitude of gradients
# magnitude = np.sqrt(grad_x**2 + grad_y**2)
# normalized_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply masks to the gradient magnitude using cv2.bitwise_and
result1 = cv2.bitwise_and(normalized_magnitude, normalized_magnitude, mask=mask1)
result2 = cv2.bitwise_and(normalized_magnitude, normalized_magnitude, mask=mask2)

# Display the results
cv2.imshow('Sobel X', grad_x)
cv2.imshow('Sobel Y', grad_y)

# cv2.imshow('Combination of Sobel X and Sobel Y', normalized_magnitude)
# cv2.imshow('Thresholded Sobel', normalized_magnitude)
combined_window = np.hstack((normalized_magnitude, thresholded))
cv2.imshow('Combined and Thresholded Sobel', combined_window)

combined_window = np.hstack((result1, result2))
cv2.imshow('Result1 and Result2', combined_window)

cv2.waitKey(0)
cv2.destroyAllWindows()
