# 设置模型、数据、训练方法等
from AI_Model_Dictionary.Model_Setting import Model_Deploy_Operation as mdo
from AI_Model_Dictionary.Data_Setting import Data_Load_Operation as dlo
from AI_Model_Dictionary.Train_Setting import Train_Method_Selection as trm
from AI_Model_Dictionary.Test_Setting import Test_Method_Selection as tem
from AI_Model_Dictionary.Loss_Setting import Loss_Value_Computation as lvc
from AI_Model_Dictionary.Optimizer_Setting import Optimizer_Deploy_Operation as odo
# 输出网络结构和性能
import torch
from thop import profile          #计算flops等model性能
from thop import clever_format
from torchsummary import summary  # 输出网络结构


# 20200719 通过指定类型加载模型
def Modelling_for_Will(model_type, param, device):
    return mdo(model_type, param, device)

# 20200719 通过指定名称加载数据
def LoadData_for_Will(dataset_type, train_batch_size, test_batch_size):
    return dlo(dataset_type, train_batch_size, test_batch_size)

# 20200719 通过指定名称选择训练方法
def Train_for_Will(train_type, model, device, train_loader, optimizer, loss, epoch, interval, batch_number, output_file, Watcher):
    return trm(train_type, model, device, train_loader, optimizer, loss, epoch, interval, batch_number, output_file, Watcher)

# 20200719 通过指定名称选择测试方法
def Test_for_Will(test_type, model, loss, device, test_loader, batch_number, output_file):
    return tem(test_type, model, loss, device, test_loader, batch_number, output_file)

# 20200719 通过指定名称选择损失函数计算方法
def Loss_for_Will(loss_type):
    return lvc(loss_type)

# 20200719 通过指定名称选择优化器部署方法
def Optim_for_Will(optimizer_type, model, lr, momentum, beta1, beta2, eps):
    return odo(optimizer_type, model, lr, momentum, beta1, beta2, eps)

def count_your_model(model, device, channel_num, sample_num, sample_X, sample_Y):
    input = torch.randn(sample_num, channel_num, sample_X, sample_Y)
    flops, params = profile(model, inputs=(input.to(device),))
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params

def summary_your_model(model, Device , sample_num, sample_X, sample_Y):
    if Device == 'cpu':
        summary(model, (sample_num, sample_X, sample_Y), device=Device)
    if Device == 'cuda':
        summary(model.cuda(), (sample_num, sample_X, sample_Y))



