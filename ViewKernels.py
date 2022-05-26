import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
torch.manual_seed(0)

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

model = CNN(out_1=160, out_2=320, out_3=480)
model.load_state_dict(torch.load("Model/Epochs50.pth"))
kernels = model.cnn1.weight.data.clone()
print(kernels.size())
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
filter_img = torchvision.utils.make_grid(kernels, nrow = 16)
plt.imshow(filter_img.permute(1, 2, 0))
plt.savefig('conv1_filters_160_kernels.png')
plt.show()