import torch
from torchvision import datasets, transforms
from DataBase_Train_Test.Coal_data.Dataset_Coal import Dataset_Coal
from DataBase_Train_Test.Stock_data.Dataset_Stock import Dataset_Stock

# 20200719 设置Data_Load_Operation函数用于数据集选择
def Data_Load_Operation(dataset_type, train_batch_size, test_batch_size):
    if dataset_type == 'MNIST':
        #提取MNIST_data的训练数据
        train_dataloader = torch.utils.data.DataLoader(
            datasets.MNIST("./DataBase_Train_Test/MNIST_data", train=True, download=False,
                        transform=transforms.Compose([
                            #transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=train_batch_size, shuffle=True,  # shuffle是否进行洗牌
            num_workers=1, pin_memory=True  # True加快训练
        )
        #提取MNIST_data的测试数据
        test_dataloader = torch.utils.data.DataLoader(
            datasets.MNIST("./DataBase_Train_Test/MNIST_data", train=False, download=False,
                        transform=transforms.Compose([
                            #transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=test_batch_size, shuffle=True,
            num_workers=1, pin_memory=True
        )
    if dataset_type == 'Coal':
        # 加载Coal数据集
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        #提取Coal的训练数据
        train_data = Dataset_Coal(txt=r'C:\Users\moonkeeper86\Desktop\AI_Dict_GPU\DataBase_Train_Test\Coal_data\image_Train.txt',
                                  transform=transforms.ToTensor(), input_resize=(28, 28))
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        #提取Coal的测试数据
        test_data = Dataset_Coal(txt=r'C:\Users\moonkeeper86\Desktop\AI_Dict_GPU\DataBase_Train_Test\Coal_data\image_Test.txt',
                                 transform=transforms.ToTensor(), input_resize=(28, 28))
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_data, batch_size=test_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    if dataset_type == 'Stock':
        # 加载Stock数据集
        stock = Dataset_Stock()
        train_dataloader = torch.utils.data.DataLoader(
            dataset=stock, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(
            dataset=stock, batch_size=test_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # print('num_of_testData:', len(test_data))
        # print('num_of_trainData:', len(train_data))
    return train_dataloader, test_dataloader
