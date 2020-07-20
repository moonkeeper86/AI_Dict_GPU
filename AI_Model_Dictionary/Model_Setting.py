
# 20200715整合网络和子模块CNN_Model_Library
from AI_Model_Dictionary import CNN_Model_Library as cnn
# 20200716 RNN_Model_Library
from AI_Model_Dictionary import RNN_Model_Library as rnn
# 20200715图像处理方法ImageProcessing_Method_Dictionary    
from AI_Model_Dictionary import ImageProcessing_Method_Dictionary as img_proc

# 20200719 设置Model_Deploy_Operation函数用于模型选择
def Model_Deploy_Operation(model_type='LeNet_5', param=[], device=[]):
    if model_type == 'LeNet_5':
        model = cnn.LeNet_5(param).to(device)
    if model_type == 'AlexNet':
        model = cnn.AlexNet(param).to(device)
    if model_type == 'NetworkinNetwork':
        model = cnn.NetworkinNetwork(param).to(device)
    if model_type == 'AlexNet':
        model = cnn.InceptionV1(param).to(device)
    if model_type == 'InceptionV1':
        model = cnn.InceptionV2(param).to(device)
    if model_type == 'InceptionV3':
        model = cnn.InceptionV3(param).to(device)
    if model_type == 'InceptionV4':
        model = cnn.InceptionV4(param).to(device)
    if model_type == 'RNN_Ori_iSequential':
        model = rnn.RNN_Ori_iSequential(
            param[0], param[1], param[2], param[3], device).to(device)
    if model_type == 'RNN_Ori_Sequential':
        model = rnn.RNN_Ori_Sequential(
            param[0], param[1], param[2], device).to(device)
    return model
