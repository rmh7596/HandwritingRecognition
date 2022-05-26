import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
torch.manual_seed(0)
train_directory = "Data/"

class Dataset(Dataset):
    def __init__(self, transform=None, training=True):
        self.dataset = []
        self.transform = transform

        for letter in os.listdir(train_directory):
            letter_dir = os.path.join(train_directory, letter)
            letter_files = [os.path.join(letter_dir,file) for file in os.listdir(letter_dir) if file.endswith(".jpg")]
            letter_files.sort()
            for sample in letter_files:
                label = ord(letter)%97 # Modularlly divides by 97 to label chars a-z with ints 0-25
                self.dataset.append([sample, label])

        # Using a 80/20 split
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=25)
        
        if training:
            self.all_files = train
            self.len = len(self.all_files)
        else: 
            self.all_files = test
            self.len = len(self.all_files)

    def getDatasetlen(self):
        return len(self.dataset)
        
    def __len__(self):
        return self.len

    def __getitem__(self, id):
        image = Image.open(self.dataset[id][0])
        label = self.dataset[id][1]

        if self.transform: # Apply a transform if needed
            image = self.transform(image)

        return image, label


class CNN(nn.Module):
    def __init__(self, out_1, out_2, out_3):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2) # Convolution layer, output is 160
        self.maxpool1=nn.MaxPool2d(kernel_size=2) # Max pooling layer, output is 80
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2) # Output is 80
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # Output is 40
        self.cnn3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=5, padding=2) # Output is 40
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) # Output is 20
        self.fc1 = nn.Linear(20*20*out_3, 26) # Fully connected neural network

    def forward(self,x):
        x = self.cnn1(x)        # Convolution
        x = torch.relu(x)       # Activation
        x = self.maxpool1(x)    # Pooling

        x = self.cnn2(x)        # Convolution
        x = torch.relu(x)       # Activation
        x = self.maxpool2(x)    # Pooling

        x = self.cnn3(x)        # Convolution
        x = torch.relu(x)       # Activation
        x = self.maxpool3(x)    # Pooling

        x = x.view(x.size(0), -1) # 1-D input
        x = self.fc1(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN(out_1=160, out_2=320, out_3=480)
model.to(device)

train_dataset = Dataset(transform=transforms.ToTensor(), training=True)
mean, std = 0.0, 0.0

for image, _ in train_dataset:
    mean += image.mean()
    std += image.std()

totalDatasetLength = train_dataset.getDatasetlen()
mean = mean/totalDatasetLength
std = std/totalDatasetLength

composed = transforms.Compose([transforms.ToTensor(), transforms.Resize([160,160]), transforms.Normalize(mean, std)])
train_dataset = Dataset(transform=composed, training=True)
validation_dataset = Dataset(transform=composed, training=False)

batch_size = 26
momentum = 0.9
lr=0.00001
num_epochs = 50

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
validationloader = DataLoader(dataset=validation_dataset, batch_size=batch_size)
n_test = len(validation_dataset)

loss_list = []
accuracy_list = []
test_images = []
predicted_label = []
actual_label = []

def train_model(n_epochs):
    for epoch in range(n_epochs):
        print(f"Epoch {epoch} started")
        for x, y in trainloader:
            x, y=x.to(device), y.to(device)
            model.train() 
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

        correct = 0
        
        for x_test, y_test in validationloader:
            x_test, y_test=x_test.to(device), y_test.to(device)
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()

            # Uncomment to collect misclassified samples
            if epoch == n_epochs-1:
                for i in range(len(x_test)):
                    if not (yhat[i]==y_test[i]):
                        test_images.append(x_test[i].cpu())
                        predicted_label.append(yhat[i].cpu())
                        actual_label.append(y_test[i].cpu())

        loss_list.append(loss.cpu().data)
        accuracy = correct / n_test
        accuracy_list.append(accuracy)

train_model(num_epochs)

PATH = "Model"
NAME = f"Epochs{num_epochs}.pth"
torch.save(model.state_dict(), os.path.join(PATH, NAME))

print("Max Accuracy %:", max(accuracy_list) * 100)
print("Length of misclassified samples", len(test_images))

def show_misclassified_samples():
    for i in range(5):
        fig, ax1 = plt.subplots()
        pred = chr(predicted_label[i].cpu().numpy() +  97)
        actual = chr(actual_label[i].cpu().numpy() + 97)
        ax1.set_title(f"Predicted value: {pred}, Actual value: {actual}")
        img = test_images[i].cpu().numpy()
        plt.imshow(np.squeeze(img), cmap="gray")
        plt.show()

def show_stats():
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(loss_list,color=color)
    ax1.set_xlabel('epoch',color=color)
    ax1.set_ylabel('total loss',color=color)
    ax1.tick_params(axis='y', color=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  
    ax2.plot( accuracy_list, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()
    plt.show()

def show_data():
    for i in range(500, 505):
        fig, ax1 = plt.subplots()
        label = chr(train_dataset[i][1] + 97)
        ax1.set_title(f"Label: {label}")
        img = train_dataset[i][0].numpy()
        plt.imshow(np.squeeze(img), cmap="gray")
        plt.show()

show_stats()
#show_data()
show_misclassified_samples()