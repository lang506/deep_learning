import pandas as pd
import torch
import torch.nn as nn
from lenet_model import LeNet
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import copy
import time

#数据处理
def train_val_data_process():
    train_data = FashionMNIST("./data",
                              train=True,
                              transform=transforms.Compose([transforms.Resize(28),
                                                            transforms.ToTensor()]),
                              download=True,
                              )
    #分割数据为测试集和验证集
    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])

    train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True,num_workers=2)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=64, shuffle=True,num_workers=2)

    return train_loader,val_loader

#模型训练
def train_model_process(model,train_loader,val_loader,num_epochs):
    # 定义使用设备
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #定义损失函数
    criterion = nn.CrossEntropyLoss()
    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model=model.to(device)
    #复制当前模型参数
    best_model_wts=copy.deepcopy(model.state_dict())

    """初始化总参数"""
    #最高准确度
    best_acc=0.0
    #训练集损失列表以及验证集损失列表
    train_loss_all=[]
    val_loss_all=[]
    #训练集准确率列表以及验证集准确率列表
    train_acc_all=[]
    val_acc_all=[]

    #当前时间
    since=time.time()
    for epoch in range(num_epochs):
        print('第{}轮'.format(epoch+1))
        print('*' * 45)

        """初始化每一轮参数"""
        #训练集损失函数、准确个数、样本数量
        train_loss=0.0
        train_acc=0.0
        train_num=0
        #验证集损失函数、准确个数、样本数量
        val_loss=0.0
        val_acc=0.0
        val_num=0

        """开始训练"""
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            #设置训练模式
            model.train()
            """前向传播"""
            #输出
            outputs = model(images)
            #计算损失
            loss = criterion(outputs, labels)
            #输出预测的行下标
            pre_lab=torch.argmax(outputs,dim=1)

            """反向传播更新参数"""
            #梯度置为0
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #更新参数
            optimizer.step()

            """训练参数评判"""
            #损失值累加
            train_loss += loss.item()*images.size(0)
            #准确数量
            train_acc += torch.sum(pre_lab==labels.data)
            #当前训练样本数量
            train_num += images.size(0)

        """开始验证"""
        for step, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            # 设置验证模式
            model.eval()

            """前向传播"""
            # 输出
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            # 输出预测的行下标
            pre_lab = torch.argmax(outputs, dim=1)

            """反向传播更新参数"""
            # 梯度置为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            """训练参数评判"""
            # 损失值累加
            val_loss += loss.item() * images.size(0)
            # 准确数量
            val_acc += torch.sum(pre_lab == labels.data)
            # 当前训练样本数量
            val_num += images.size(0)

        """计算并保存每一轮训练以及验证的Loss值以及准确率"""
        #训练集中
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_acc.double().item()/train_num)
        #验证集中
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_acc.double().item()/val_num)

        print("训练轮次:{} 训练损失为:{:.4f},训练集正确率为:{:.4f}".format(epoch+1,train_loss_all[-1],train_acc_all[-1]))
        print("训练轮次:{} 验证集损失为:{:.4f},验证集正确率为:{:.4f}".format(epoch+1,val_loss_all[-1], val_acc_all[-1]))

        #保存最优模型
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        #计算耗时
        cost_time = time.time() - since
        print("训练和验证所花费的时间: {:.2f} 秒".format(cost_time))

    #训练以及验证完成后保存模型
    torch.save(best_model_wts,'lenet_model.pth')

    train_process=pd.DataFrame(data={
        "epoch":range(num_epochs),
        "train_loss_all":train_loss_all,
        "train_acc_all":train_acc_all,
        "val_loss_all":val_loss_all,
        "val_acc_all":val_acc_all
    })
    return train_process

def plot_acc_loss(train_process):
    plt.figure(figsize=(12, 8))

    #绘制损失数值
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'],train_process.train_loss_all, "ro-",label='train loss')
    plt.plot(train_process['epoch'],train_process.val_loss_all, "bs-",label='val loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    #绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'],train_process.train_acc_all, "ro-",label='train accuracy')
    plt.plot(train_process['epoch'],train_process.val_acc_all, "bs-",label='val accuracy')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.show()

if __name__ == '__main__':
    #加载模型
    LeNet=LeNet()
    #加载数据
    train_loader, val_loader=train_val_data_process()
    #开始训练
    train_process=train_model_process(LeNet,train_loader,val_loader,20)
    #绘制图像
    plot_acc_loss(train_process)




        













