import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    
])


dataLoc = "../Year/2017crop"
path = '../Models/2017Model.pth'
mapping = "../Models/2017Map.csv"

dataset = ImageFolder(root=dataLoc, transform=data_transform)
class_to_idx_mapping = dataset.class_to_idx
idx_to_class_mapping = {idx: class_name for class_name, idx in class_to_idx_mapping.items()}

print("Index to Class Mapping:")
for idx, class_name in idx_to_class_mapping.items():
    print(f"{idx}: {class_name}")

with open(mapping, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Class Name', 'Index'])
    
    for class_name, idx in class_to_idx_mapping.items():
        csv_writer.writerow([class_name, idx])


total_size = len(dataset)
train_size = int(0.7 * total_size)  # 70 trainin
test_size = (total_size - train_size) // 2  # 15 testing
val_size = total_size - train_size - test_size  # 15 val

# Use random_split to partition the dataset
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

# Create DataLoaders
# Dataloaders are an internal thing for pytorch to work
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)


resnet50 = models.resnet50(pretrained=False).to(device) 
num_classes = len(dataset.classes)
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes).to(device)  

#Adams actualy works good here!
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.0005)

# Training loop
epochs = 15
for epoch in range(epochs):
    resnet50.train()
    total_loss = 0.0
    total_samples = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  
        optimizer.zero_grad()
        output = resnet50(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        total_samples += target.size(0)

    average_loss = total_loss / total_samples

    print(f'Epoch {epoch + 1}, Average Loss: {average_loss}')



resnet50.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)  
        output = resnet50(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

sample_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
sample_images, sample_labels = next(iter(sample_loader))


sample_images = sample_images.to(device)
resnet50.eval()
with torch.no_grad():
    sample_outputs = resnet50(sample_images)


_, predicted_classes = torch.max(sample_outputs, 1)


#write output table
num_images_to_display = 25
num_images_per_row = 5  
num_rows = (num_images_to_display + num_images_per_row - 1) // num_images_per_row

#Plot
for i in range(num_images_to_display):
    image = sample_images[i].cpu().numpy().transpose((1, 2, 0))
    true_label = sample_labels[i].item()
    predicted_label = predicted_classes[i].item()

    #To make it not cramped
    row_num = i // num_images_per_row
    col_num = i % num_images_per_row

    plt.subplot(num_rows, num_images_per_row, i + 1)
    plt.imshow(image)
    plt.title(f'True: {true_label}, Predicted: {predicted_label}')
    plt.axis('off')

plt.show()

#Way to save it, really easy
torch.save(resnet50, path)
