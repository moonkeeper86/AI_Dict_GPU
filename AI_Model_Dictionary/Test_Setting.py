import torch
import torch.nn.functional as F
import numpy as np
import time
import tensorwatch as tw


def Test_Method_Selection(test_type, model, loss, device, test_loader, batch_number, output_file):
    model.eval()
    total_loss = 0.  # 初始总偏差
    correct = 0.  # 初始准确率
    t1 = []
    it_num = 1
    if test_type == 'normal_test':
        with torch.no_grad():
            # 序号乱序并取指定数量样本
            idx_list = np.arange(0, len(test_loader.dataset.data), 1)
            np.random.shuffle(idx_list)
            idx_list = idx_list[0:batch_number]
            for it_num in range(0, int(batch_number/test_loader.batch_size)):
                if t1 == []:
                    t1 = time.time()
                # it_s,it_e为每轮读取图片的idx_list序号
                it_s = it_num*test_loader.batch_size
                it_e = (it_num+1)*test_loader.batch_size
                if it_s < len(test_loader.dataset.data) and it_e < len(test_loader.dataset.data):
                    # 01 读取批量数据
                    sp = np.shape(
                        test_loader.dataset.data[idx_list[it_s:it_e], :, :])
                    data = test_loader.dataset.data[idx_list[it_s:it_e], :, :].view(
                        [sp[0], -1, sp[1], sp[2]]).float()
                    target = test_loader.dataset.targets[idx_list[it_s:it_e]]
                    #data = (data - test_loader.dataset.data.float().mean())/test_loader.dataset.data.float().std()  # mean()和std()需要在计算前对数据做float转换
                    data, target = data.to(device), target.to(device)
                    #02 利用当前模型model对输入数据data进行输出
                    output = model(data)
                    #03 计算输出结果output与实际标签target之间的偏差作为累加为总损失total_loss
                    total_loss += loss(output,target)
                    #04 独热码值输出决策类别，上面的output为多维向量，为各类决策的概率，独热码则是最高概率为决策结果
                    pred = output.argmax(dim=1)
                    #05 独热output如与target相同则正确
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    #07 每interval次优化输出一次
                    if it_num % 1 == 0:
                        t2 = time.time()
                        print("Test Iterantion: [{:4d}/{:4d} @ {:3.6f}%], Total_Loss: {:6f}, Traintime: {:4.6f}s".format(
                            it_num+1, int(batch_number/test_loader.batch_size), it_num/int(batch_number/test_loader.batch_size), total_loss, t2-t1))
                        if output_file != []:
                            output_file.log_to_file("Test Iterantion: [{:4d}/{:4d} @ {:3.6f}%], Total_Loss: {:6f}, Traintime: {:4.6f}s".format(
                                it_num+1, int(batch_number/test_loader.batch_size), it_num/int(batch_number/test_loader.batch_size), total_loss, t2-t1))
                        t1 = []
                    it_num += 1
                else:
                    break
            total_loss /= (it_num+1)*test_loader.batch_size
            acc = correct/((it_num+1)*test_loader.batch_size)*100
            print("Test loss: {}, Accuracy: {}%".format(total_loss, acc))
            if output_file != []:
                output_file.log_to_file(
                    "Test loss: {}, Accuracy: {}%".format(total_loss, acc))
    if test_type == 'series_test':
        with torch.no_grad():
            # 序号乱序并取指定数量样本
            idx_list = np.arange(0, len(test_loader.dataset.data), 1)
            np.random.shuffle(idx_list)
            idx_list = idx_list[0:batch_number]
            for it_num in range(0, int(batch_number/test_loader.batch_size)):
                if t1 == []:
                    t1 = time.time()
                # it_s,it_e为每轮读取图片的idx_list序号
                it_s = it_num*test_loader.batch_size
                it_e = (it_num+1)*test_loader.batch_size
                if it_s < len(test_loader.dataset.data) and it_e < len(test_loader.dataset.data):
                    # 01 读取批量数据
                    data = test_loader.dataset.data[idx_list[it_s:it_e], :]
                    target = test_loader.dataset.targets[idx_list[it_s:it_e]]
                    data = data.astype(np.float32)
                    target = target.astype(np.float32)
                    data, target = torch.from_numpy(data).to(
                        device), torch.from_numpy(target).to(device)
                    #02 利用当前模型model对输入数据data进行输出
                    data = (data-test_loader.dataset.data.mean()) / \
                        test_loader.dataset.data.std()
                    output = model(data)
                    output = output*test_loader.dataset.data.std()+test_loader.dataset.data.mean()
                    #03 计算输出结果output与实际标签target之间的偏差作为累加为总损失total_loss
                    total_loss += loss(output,target)
                    #04 独热码值输出决策类别，上面的output为多维向量，为各类决策的概率，独热码则是最高概率为决策结果
                    #pred = output.argmax(dim=1)  #TODO 需要考虑pred还要不要
                    #05 独热output如与target相同则正确
                    tmp = output-target.view([test_loader.batch_size])
                    correct = tmp.square().sqrt().sum()
                    #07 每interval次优化输出一次
                    if it_num % 1 == 0:
                        t2 = time.time()
                        print("Test Iterantion: [{:4d}/{:4d} @ {:3.6f}%], Total_Loss: {:6f}, Traintime: {:4.6f}s".format(
                            it_num+1, int(batch_number/test_loader.batch_size), it_num/int(batch_number/test_loader.batch_size), total_loss, t2-t1))
                        if output_file != []:
                            output_file.log_to_file("Test Iterantion: [{:4d}/{:4d} @ {:3.6f}%], Total_Loss: {:6f}, Traintime: {:4.6f}s".format(
                                it_num+1, int(batch_number/test_loader.batch_size), it_num/int(batch_number/test_loader.batch_size), total_loss, t2-t1))
                        t1 = []
                    it_num += 1
                else:
                    break
            total_loss /= (it_num+1)*test_loader.batch_size
            acc = correct/((it_num+1)*test_loader.batch_size)
            print("Test loss: {}, Accuracy: {}%".format(total_loss, acc))
            if output_file != []:
                output_file.log_to_file(
                    "Test loss: {}, Accuracy: {}%".format(total_loss, acc))
    return total_loss, acc, data.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
