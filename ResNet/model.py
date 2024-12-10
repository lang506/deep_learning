import torch
from torch import nn
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels,use_1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.relu=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3,padding=1)
        if use_1conv:
            self.conv3=nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(output_channels)
        self.bn2=nn.BatchNorm2d(output_channels)

    def forward(self, x):
        y=self.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3 :
            x=self.conv3(x)
        y=self.relu(y+x)
        return y

class ResNet18(nn.Module):
    def __init__(self,ResidualBlock):
        super(ResNet18,self).__init__()
        self.b1=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2=nn.Sequential(
            ResidualBlock(input_channels=64, output_channels=64,use_1conv=False, strides=1),
            ResidualBlock(input_channels=64, output_channels=64,use_1conv=False, strides=1)
        )
        self.b3=nn.Sequential(
            ResidualBlock(input_channels=64, output_channels=128,use_1conv=True, strides=2),
            ResidualBlock(input_channels=128, output_channels=128,use_1conv=False, strides=1)
        )
        self.b4=nn.Sequential(
            ResidualBlock(input_channels=128, output_channels=256,use_1conv=True, strides=2),
            ResidualBlock(input_channels=256, output_channels=256,use_1conv=False, strides=1)
        )
        self.b5=nn.Sequential(
            ResidualBlock(input_channels=256, output_channels=512,use_1conv=True, strides=2),
            ResidualBlock(input_channels=512, output_channels=512,use_1conv=False, strides=1)
        )
        self.b6=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x=self.b1(x)
        x=self.b2(x)
        x=self.b3(x)
        x=self.b4(x)
        x=self.b5(x)
        x=self.b6(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(ResidualBlock).to(device)
    print(summary(model, (1, 224, 224)))






















