import torch
import torch.nn.functional as F
import numpy as np
import time
import tensorwatch as tw

def Train_Method_Selection(train_type, model, device, train_loader, optimizer, loss, epoch, interval, batch_number, output_file, Watcher):
    loss_val = []
    idx_val = []
    model.train()  # 必备，将模型设置为训练模式
    t1 = []
    # 序号乱序并取指定数量样本
    idx_list = np.arange(0, len(train_loader.dataset.data), 1)
    np.random.shuffle(idx_list)
    idx_list = idx_list[0:batch_number]
    if train_type == 'normal_train':
        for it_num in range(0, int(batch_number/train_loader.batch_size)):
            if t1 == []:
                t1 = time.time()
            # it_s,it_e为每轮读取图片的idx_list序号
            it_s = it_num*train_loader.batch_size
            it_e = (it_num+1)*train_loader.batch_size
            if it_s < len(train_loader.dataset.data) and it_e < len(train_loader.dataset.data):
                #01 读取批量数据
                sp = np.shape(train_loader.dataset.data[idx_list[it_s:it_e], :, :])
                data = train_loader.dataset.data[idx_list[it_s:it_e], :, :].view(
                    [sp[0], -1, sp[1], sp[2]]).float()
                target = train_loader.dataset.targets[idx_list[it_s:it_e]]
                #data = (data - train_loader.dataset.data.float().mean())/train_loader.dataset.data.float().std()  # mean()和std()需要在计算前对数据做float转换
                data, target = data.to(device), target.to(device)
                #02 利用当前模型model对输入数据data进行预测
                pred = model(data)
                #03 计算预测结果pred与实际标签target之间的偏差作为损失los
                if len(np.shape(pred)) == 1:
                    los0 = pred[0]
                    los0 = loss(los0, target)
                    los1 = pred[1]
                    los1 = loss(los1, target)
                    los2 = pred[2]
                    los2 = loss(los2, target)
                    los = los0 + los1 + los2
                if len(np.shape(pred)) == 2:
                    los = loss(pred, target)
                #04 梯度清零，清空优化器
                optimizer.zero_grad()
                #05 误差los反向传播
                los.backward()
                #06 模型优化，调整参数
                optimizer.step()
                #202006172233 利用Watcher()建立观测器观测iter和los
                Watcher.observe(iter=it_num, loss=los)
                time.sleep(1)
                #07 每interval次优化输出一次
                if it_num % interval == 0:
                    t2 = time.time()
                    print("Train Epoch: {:4d}th, iterantion: [{:4d}/{:4d} @ {:3.6f}%], Loss: {:6f}, Traintime: {:4.6f}s".format(
                        epoch+1, it_num, int(batch_number/train_loader.batch_size), it_num/int(batch_number/train_loader.batch_size), los.item(), t2-t1))
                    if output_file != []:
                        output_file.log_to_file("Train Epoch: {:4d}th, iterantion: [{:4d}/{:4d} @ {:3.6f}%], Loss: {:6f}, Traintime: {:4.6f}s".format(
                            epoch+1, it_num, int(batch_number/train_loader.batch_size), it_num/int(batch_number/train_loader.batch_size), los.item(), t2-t1))
                    idx_val.append(it_num)
                    loss_val.append(los.item())
                    t1 = []
                it_num += 1
            else:
                break
    if train_type == 'series_train':
        for it_num in range(0, int(batch_number/train_loader.batch_size)):
            if t1 == []:
                t1 = time.time()
            # it_s,it_e为每轮读取图片的idx_list序号
            it_s = it_num*train_loader.batch_size
            it_e = (it_num+1)*train_loader.batch_size
            if it_s < len(train_loader.dataset.data) and it_e < len(train_loader.dataset.data):
                #01 读取批量数据
                data = train_loader.dataset.data[idx_list[it_s:it_e], :]
                target = train_loader.dataset.targets[idx_list[it_s:it_e]]
                data = data.astype(np.float32)
                target = target.astype(np.float32)
                data, target = torch.from_numpy(data).to(
                    device), torch.from_numpy(target).to(device)
                #02 利用当前模型model对输入数据data进行预测
                data = (data-train_loader.dataset.data.mean())/train_loader.dataset.data.std()
                pred = model(data)
                pred = pred*train_loader.dataset.data.std()+train_loader.dataset.data.mean()
                #03 计算预测结果pred与实际标签target之间的偏差作为损失los
                if len(np.shape(pred)) == 1:
                    los0 = pred[0]
                    los0 = loss(los0, target.long())
                    los1 = pred[1]
                    los1 = loss(los1, target.long())
                    los2 = pred[2]
                    los2 = loss(los2, target.long())
                    los = los0 + los1 + los2
                if len(np.shape(pred)) == 2:
                    los = loss(pred, target)
                #04 梯度清零，清空优化器
                optimizer.zero_grad()
                #05 误差los反向传播
                los.backward()
                #06 模型优化，调整参数
                optimizer.step()
                #202006172233 利用Watcher()建立观测器观测iter和los
                Watcher.observe(iter=it_num, loss=los)
                time.sleep(1)
                #07 每interval次优化输出一次
                if it_num % interval == 0:
                    t2 = time.time()
                    print("Train Epoch: {:4d}th, iterantion: [{:4d}/{:4d} @ {:3.6f}%], Loss: {:6f}, Traintime: {:4.6f}s".format(
                        epoch+1, it_num, int(batch_number/train_loader.batch_size), it_num/int(batch_number/train_loader.batch_size), los.item(), t2-t1))
                    if output_file != []:
                        output_file.log_to_file("Train Epoch: {:4d}th, iterantion: [{:4d}/{:4d} @ {:3.6f}%], Loss: {:6f}, Traintime: {:4.6f}s".format(
                            epoch+1, it_num, int(batch_number/train_loader.batch_size), it_num/int(batch_number/train_loader.batch_size), los.item(), t2-t1))
                    idx_val.append(it_num)
                    loss_val.append(los.item())
                    t1 = []
                it_num += 1
            else:
                break
    return idx_val, loss_val
