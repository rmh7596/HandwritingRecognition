import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
torch.manual_seed(0)
train_directory = "Data/"

class Dataset(Dataset):
    def __init__(self, transform=None, train=True):
        self.all_files = []
        self.labels = []
        self.transform = transform

        for letter in os.listdir(train_directory):
            letter_dir = os.path.join(train_directory, letter)
            letter_files = [os.path.join(letter_dir,file) for file in os.listdir(letter_dir) if file.endswith(".jpg")]
            letter_files.sort()
            for sample in letter_files:
                label = ord(letter)%97 # Modularlly divides by 97 to label chars a-z with ints 0-25
                self.labels.append(label)
                self.all_files.append(sample)
        
        random.seed(1)
        random.shuffle(self.all_files)
        random.shuffle(self.labels)
        # Shuffle the order of the images
        # Using a 80/20 split
        if train:
            self.all_files = self.all_files[0:8335]
            self.labels = self.labels[0:8335]
            self.len = len(self.all_files)
        else: 
            self.all_files = self.all_files[8335::]
            self.labels = self.labels[8335::]
            self.len = len(self.all_files)

        
    def __len__(self):
        return self.len

    def __getitem__(self, id):
        image = Image.open(self.all_files[id])
        label = self.labels[id]

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
        #self.cnn3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=5, padding=2) # Output is 20
        #self.maxpool3 = nn.MaxPool2d(kernel_size=2) # Output is 10
        self.fc1 = nn.Linear(40*40*out_2, 26) # Fully connected neural network

    def forward(self,x):
        x = self.cnn1(x)        # Convolution
        x = torch.relu(x)       # Activation
        x = self.maxpool1(x)    # Pooling

        x = self.cnn2(x)        # Convolution
        x = torch.relu(x)       # Activation
        x = self.maxpool2(x)    # Pooling

        #x = self.cnn3(x)        # Convolution
        #x = torch.relu(x)       # Activation
        #x = self.maxpool3(x)    # Pooling

        x = x.view(x.size(0), -1) # 1-D input
        x = self.fc1(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN(out_1=160, out_2=80, out_3=480)
model.to(device)

train_dataset = Dataset(transform=transforms.ToTensor(), train=True)
mean, mean_squared = 0.0, 0.0

for image, _ in train_dataset:
    mean += image[0].mean()
    mean_squared += torch.mean(image**2)

mean = mean/len(train_dataset)
#std = sqrt(E[X^2] - (E[X])^2)
std = (mean_squared / len(train_dataset) - mean ** 2) ** 0.5

composed = transforms.Compose([transforms.ToTensor(), transforms.Resize([160,160]), transforms.Normalize(mean, std)])
train_dataset = Dataset(transform=composed, train=True)
validation_dataset = Dataset(transform=composed, train=False)

batch_size = 52
momentum = 0.9
lr=0.000001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
validationloader = DataLoader(dataset=validation_dataset, batch_size=batch_size)
n_test = len(validation_dataset)

loss_list = []
accuracy_list = []
test_image = []
predicted_label = []
actual_label = []
sample_number = []

def train_model(n_epochs):
    correct = 0
    for epoch in range(n_epochs):
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

            for i in range(len(x_test)):
                if not (yhat[i]==y_test[i]):
                    sample_number.append(i)
                    test_image.append(x_test[i])
                    predicted_label.append(y_test[i])
                    actual_label.append(yhat[i])

        loss_list.append(loss.cpu().data)
        accuracy = correct / n_test
        accuracy_list.append(accuracy)

    return correct

print("Num correct: ", train_model(1))
torch.cuda.empty_cache()
print("Accuracy %", max(accuracy_list) * 100)

def show_misclassified_sample():
    fig, ax1 = plt.subplots()
    pred = predicted_label[0].cpu().numpy()
    actual = predicted_label[0].cpu().numpy()
    ax1.set_title(f"Predicted value: {pred}, Actual value: {actual}")
    img = test_image[0].cpu().numpy()
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

show_misclassified_sample()