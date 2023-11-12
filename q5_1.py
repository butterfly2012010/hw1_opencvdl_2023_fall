from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from pathlib import Path

# Define the directory where images are stored
image_dir = Path("E:/YN/opencvdl/Hw1/Dataset_OpenCvDl_Hw1/Q5_image/Q5_1/")

# Check if the directory exists and list all image files
if image_dir.is_dir():
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg'))
else:
    image_files = []

# Define the transformations
transformations = {
    'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=1),
    'RandomVerticalFlip': transforms.RandomVerticalFlip(p=1),
    'RandomRotation': transforms.RandomRotation(30)
}

# Load images and apply transformations
augmented_images = {}
for image_file in image_files:
    image_name = image_file.name
    img = Image.open(image_file)
    augmented_images[image_name] = {}
    for transform_name, transform in transformations.items():
        augmented_images[image_name][transform_name] = transform(img)

# augmented_images

#####################################################################################################


# # Create a figure with subplots
# fig, axs = plt.subplots(3, 3, figsize=(6, 6))

# # Flatten the array of axes for easy iteration
# axs = axs.flatten()

# # Loop over the first nine images and transformations and plot
# for i, image_file in enumerate(image_files[:9]):
#     img = Image.open(image_file)
#     axs[i].imshow(img)
#     axs[i].set_title(f'Original Image {i+1}')
#     axs[i].axis('off')

# plt.show()

# Apply transformations and plot
# fig, axs = plt.subplots(len(transformations), 3, figsize=(6, 6))
fig, axs = plt.subplots(3, 3, figsize=(6, 6))

# for i, transform in enumerate(transformations.values()):
#     for j, image_file in enumerate(image_files):  # apply each transformation to three images
#         img = Image.open(image_file)
#         axs[i, j].imshow(transform(img))
#         axs[i, j].set_title(f'Augmentation {i+1} on Image {j+1}')
#         axs[i, j].axis('off')

# Flatten the array of axes for easy iteration
axs = axs.flatten()

for i, image_file in enumerate(image_files):
    img = Image.open(image_file)
    img = transformations['RandomHorizontalFlip'](img)
    img = transformations['RandomVerticalFlip'](img)
    img = transformations['RandomRotation'](img)
    axs[i].imshow(img)
    axs[i].set_title(os.path.split(image_file)[1].split('.')[0])
    axs[i].axis('off')


plt.show()