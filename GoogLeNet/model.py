import torch
from torch import nn
from torchsummary import summary


class Inception(nn.Module):
    def __init__(self, input_channel,output_channel1,output_channel2,output_channel3,output_channel4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU(inplace=True)
        #path1
        self.path1=nn.Conv2d(in_channels=input_channel,out_channels=output_channel1,kernel_size=1)

        #path2
        self.path2_1=nn.Conv2d(in_channels=input_channel,out_channels=output_channel2[0],kernel_size=1)
        self.path2_2=nn.Conv2d(in_channels=output_channel2[0],out_channels=output_channel2[1],kernel_size=3,padding=1)

        #path3
        self.path3_1=nn.Conv2d(in_channels=input_channel,out_channels=output_channel3[0],kernel_size=1)
        self.path3_2=nn.Conv2d(in_channels=output_channel3[0],out_channels=output_channel3[1],kernel_size=5,padding=2)

        #path4
        self.path4_1=nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.path4_2=nn.Conv2d(in_channels=input_channel,out_channels=output_channel4,kernel_size=1)

    def forward(self,x):
        p1=self.ReLU(self.path1(x))
        p2=self.ReLU(self.path2_2(self.ReLU(self.path2_1(x))))
        p3=self.ReLU(self.path3_2(self.ReLU(self.path3_1(x))))
        p4=self.ReLU(self.path4_2(self.path4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim=1)

class GoogLeNet(nn.Module):
    def __init__(self,Inception):
        super(GoogLeNet, self).__init__()
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)  # 内置的 LRN 层
        self.b1=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            self.lrn,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            self.lrn,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x=self.b1(x)
        x=self.b2(x)
        x=self.b3(x)
        x=self.b4(x)
        x=self.b5(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=GoogLeNet(Inception).to(device)

    print(summary(model,(1,224,224)))


















