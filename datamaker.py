from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt

# Define your data transformations
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # Add more transformations as needed
])

# Load the dataset
dataset = datasets.ImageFolder(root='dataset', transform=data_transform)

# Define the sizes of your training, test, and validation sets
total_size = len(dataset)
train_size = int(0.7 * total_size)  # 70% for training
test_size = (total_size - train_size) // 2  # 15% for testing
val_size = total_size - train_size - test_size  # 15% for validation

# Use random_split to partition the dataset
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

print(train_loader)

first_train_sample = train_dataset[0]
print("First Training Sample:")
print("Class Label:", first_train_sample[1])
plt.imshow(first_train_sample[0].permute(1, 2, 0).numpy())
plt.show()

# Print the first sample from the validation dataset
first_val_sample = val_dataset[0]
print("\nFirst Validation Sample:")
print("Class Label:", first_val_sample[1])
plt.imshow(first_val_sample[0].permute(1, 2, 0).numpy())
plt.show()

# Print the first sample from the testing dataset
first_test_sample = test_dataset[0]
print("\nFirst Testing Sample:")
print("Class Label:", first_test_sample[1])
plt.imshow(first_test_sample[0].permute(1, 2, 0).numpy())
plt.show()
