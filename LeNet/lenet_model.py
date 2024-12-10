import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=10),
        )
    def forward(self, x):
        x=self.model(x)
        return x

if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=LeNet().to(device)
    print(summary(model,(1,28,28)))