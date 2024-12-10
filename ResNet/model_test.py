import torch
from model import ResidualBlock,ResNet18
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data


#数据处理
def test_data_process():
    test_data = FashionMNIST("./data",
                              train=False,
                              transform=transforms.Compose([transforms.Resize(224),
                                                            transforms.ToTensor()]),
                              download=True,
                              )
    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True,num_workers=0)
    return test_loader

def test_model_process(model,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)

    #初始化参数
    test_corrects = 0.0
    test_num = 0

    with torch.no_grad():
        for image, target in test_loader:
            #设置设备
            image = image.to(device)
            target = target.to(device)
            #进入测试模式
            model.eval()
            #输出
            output = model(image)
            #评估模型的参数设置
            pre_label=torch.argmax(output,dim=1)
            test_corrects += torch.sum(pre_label==target)
            test_num += image.size(0)
        #计算准确率
        test_accuracy = test_corrects.double().item()/test_num
        print("测试集上的准确率为{}".format(test_accuracy))

if __name__ == '__main__':
    #加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ResNet18(ResidualBlock).to(device)
    model.load_state_dict(torch.load("./ResNet18_model.pth", map_location=device,weights_only=True))
    #加载数据
    test_loader=test_data_process()
    #模型测试
    # test_model_process(model, test_loader)

    classes=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for image,target in test_loader:
            image=image.to(device)
            target=target.to(device)
            model.eval()

            output = model(image)
            pre_label=torch.argmax(output,dim=1)

            print("真实类别{}，预测类别{}".format(classes[target.item()],classes[pre_label.item()]))
















