import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation  #, RandomCrop, ColorJitter, RandomRotation
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

# NUM_EPOCHS = 40 # Number of epochs to train the model

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # RTX 2080 SUPER

# # Define transformations for the CIFAR10 dataset
# # Applying data augmentation only to the training dataset
# train_transform = transforms.Compose([
#     RandomHorizontalFlip(p=0.5),  # p=0.5 means that the transformation has a 50% probability of being applied
#     RandomVerticalFlip(p=0.5),  # p=0.5 means that the transformation has a 50% probability of being applied
#     RandomRotation(30),  # Rotates the image by a maximum of 30 degrees
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # For the test dataset, we only apply the necessary transformations
# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Load CIFAR10 dataset with the transformations
# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# # Data loaders
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# VGG19 with Batch Normalization
model = models.vgg19_bn(pretrained=False)
model.classifier[6] = nn.Linear(4096, 10) # Adjusting for CIFAR10's 10 classes
model = model.to(device)

# Print the model summary
summary(model, (3, 32, 32))  # CIFAR10 images have a size of 32x32 with 3 color channels

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# # Placeholder for training/validation loss and accuracy
# training_loss = []
# validation_loss = []
# training_accuracy = []
# validation_accuracy = []

# # Function to train and validate the model
# def train_model(model, criterion, optimizer, num_epochs=10):
#     # for epoch in range(num_epochs):
#     for epoch in tqdm(range(num_epochs), desc="Epochs"):
#         model.train()
#         running_loss = 0.0
#         running_corrects = 0

#         # Training loop
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             _, preds = torch.max(outputs, 1)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(train_dataset)
#         epoch_acc = running_corrects.double() / len(train_dataset)

#         training_loss.append(epoch_loss)
#         training_accuracy.append(epoch_acc.cpu().numpy())

#         model.eval()
#         val_running_loss = 0.0
#         val_running_corrects = 0

#         # Validation loop
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_running_loss += loss.item() * inputs.size(0)
#                 _, preds = torch.max(outputs, 1)
#                 val_running_corrects += torch.sum(preds == labels.data)

#         val_epoch_loss = val_running_loss / len(test_dataset)
#         val_epoch_acc = val_running_corrects.double() / len(test_dataset)

#         validation_loss.append(val_epoch_loss)
#         validation_accuracy.append(val_epoch_acc.cpu().numpy())

#         # print(f'Epoch {epoch+1}/{num_epochs} - '
#             #   f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
#             #   f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
#         tqdm.write(f'Epoch {epoch+1}/{num_epochs} - '
#                    f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
#                    f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

#     return model

# # Train the model
# model_trained = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)

# # Save the trained model
# torch.save(model_trained.state_dict(), "vgg19_bn_cifar10.pth")

# # Plot and save training/validation loss and accuracy
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(training_loss, label='Training Loss')
# plt.plot(validation_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(training_accuracy, label='Training Accuracy')
# plt.plot(validation_accuracy, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.savefig("training_validation_loss_accuracy.jpg")
# plt.show()
