from other.network_and_computing_record import Network_And_Computing
import sys
sys.path.append("../..")
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from VGG.mydefine_VGG16 import VGG_model

# 时间估计
def get_prediction_time(datanode_num = 0, index = 0, length = 0, cross_layer = 1, computing_a = [],
                        computing_b = [], network_state = [], input_param = [], c_out = 0):
    input_number, c_in, height, width = input_param
    if c_out == 0:
        c_out = c_in
    else:
        c_out = c_out
    kernel = 3
    # 计算 FLOPs
    FLOPs = input_number * 2 * height * length * c_out * (kernel * kernel * c_in + 1)
    # 计算时间
    comp_time = computing_a[index] * FLOPs + computing_b[index]
    # 通信开销
    comm_data = input_number * c_in * height * 4.0 / network_state[index]
    comm_time = 0
    # 判断是否是边界
    if index == 0 or index == datanode_num - 1:
        comm_time = comm_data * (cross_layer + 2 * length)
    # 中间情况
    else:
        comm_time = 2 * comm_data * (cross_layer + length)
    prediction_time = comp_time + comm_time
    return prediction_time



# #############################################################################################################
# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_network(original_tensor, datanode_num = 1, cross_layer = 1,
                                        computing_power = [], computing_a = [], computing_b = [], network_state = [], c_out = 0):
    # 优化步长
    step = 1
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    input_param = []
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        input_number, c_in, height, width = original_tensor.size()
        input_param.append(input_number)
        input_param.append(c_in)
        input_param.append(height)
        input_param.append(width)
        # 提前计算求和
        total_computing_power = 0
        for i in range(datanode_num):
            total_computing_power += computing_power[i]
        sum_computing_power = []
        for i in range(datanode_num + 1):
            sum_computing_power.append(sum(computing_power[0 : i]))

        # 定义划分长度
        length = []
        # 时间开销
        prediction_time = []
        for i in range(datanode_num):
            length.append(0)
            prediction_time.append(0)
        for it in range(datanode_num):
            length[it] = int(sum_computing_power[it+1]/total_computing_power * width) - \
                         int(sum_computing_power[it] / total_computing_power * width)
        for it in range(datanode_num):
            prediction_time[it] = get_prediction_time(datanode_num = datanode_num, index = it, length = length[it], cross_layer = cross_layer, computing_a = computing_a,
                        computing_b = computing_b, network_state = network_state, input_param=input_param, c_out = c_out)
        iter = 0
        iter_stop = 30
        diff = 0
        # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
        while(True):
            iter += 1
            # 判断是否到轮次上限
            if iter == iter_stop:
                break
            # 找出时间最值及下标
            max_value = max(prediction_time)
            min_value = min(prediction_time)
            index_max = prediction_time.index(max_value)
            index_min = prediction_time.index(min_value)
            last_diff = diff
            diff = max_value - min_value
            # 判断退出条件
            if ( diff < 0.02 or min(length) <= 2):
                break
            last_diff = diff
            length[index_max] -= step
            length[index_min] += step
            # 出错
            prediction_time[index_max] = get_prediction_time(datanode_num = datanode_num, index = index_max, length = length[index_max], cross_layer = 1, computing_a = computing_a,
                        computing_b = computing_b, network_state = network_state, input_param=input_param, c_out = c_out)
            prediction_time[index_min] = get_prediction_time(datanode_num = datanode_num, index = index_min, length = length[index_min], cross_layer = 1, computing_a = computing_a,
                        computing_b = computing_b, network_state = network_state, input_param=input_param, c_out = c_out)
            print (length)
            print (prediction_time)
        print(length)
        print(prediction_time)
        # 已经得到length，根据length确定划分范围
        start = 0
        end = 0
        for it in range(datanode_num):
            end = start + length[it]
            print("[ %d, %d]" % (start, end))
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start : int(end + cross_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer) : end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer) : int(end + cross_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
            # 更换起始位置。
            start = end
    # 返回最终的结果
    return divided_tensor, divide_record


if __name__ == "__main__":
    datanode_num = 4
    # 获得 datanode 设备的 计算能力 和 通信能力
    network_and_computing = Network_And_Computing()
    computing_power = network_and_computing.get_computing_power(datanode_num)
    network_state = network_and_computing.get_network_state(datanode_num)
    computing_a = network_and_computing.get_computing_a(datanode_num)
    computing_b = network_and_computing.get_computing_b(datanode_num)

    # 设定输入
    width = 224
    original_tensor = torch.rand(1, 64, width, width)
    cross_layer = 1

    a, b = tensor_divide_by_computing_and_network(original_tensor, datanode_num=datanode_num, cross_layer = cross_layer,
                                                  computing_power = computing_power, computing_a = computing_a,
                                                  computing_b = computing_b, network_state = network_state, c_out = 128)

