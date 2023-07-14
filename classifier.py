import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HidenConv()
        # if change the hidenconv, change the following fc1 and fc2
        self.fc1 = nn.Linear(100, 64)
        self.relu1 = nn.ReLU()
        #Test different activation function
        # self.relu1 = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 1)# last days of the quarter try your best :)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        y = self.fc1(z)
        y = self.relu1(y)
        y = self.fc2(y)
        return y, mu, log_var

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        #Test different activation functions
        # self.activation = nn.ReLU()
        self.activation = nn.LeakyReLU(0.2)
        # self.activation = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class HidenConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBN(3, 32)
        self.conv2 = ConvBN(32, 64)
        self.conv3 = ConvBN(64, 128)
        # change 256 to 100
        self.conv4 = ConvBN(128, 100)
        self.conv5 = ConvBN(100, 512)
        self.fc_mu = nn.Linear(512 * 6 * 5, 100)
        self.fc_log_var = nn.Linear(512 * 6 * 5, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

#to calculate the mu and log_var for kld loss
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

