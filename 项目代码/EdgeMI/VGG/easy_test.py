import sys
sys.path.append("../..")
sys.path.append("..")

import torch, time
import torch.nn as nn
import numpy as np
from VGG.mydefine_VGG16 import VGG_model

# 测试 计算时间和通信时间
datanode_num = 5
in_chanel = 64
cross_layer = 1
original_tensor = torch.rand(1, in_chanel, 224, 224)
out_chanel = 64
kernerl_size = 3
# com_k值越小，算力越高，斜率的倒数表示算力,选择合适的算力机器，设置合适的带宽

computing_power = [ [0.06245, 0.06207, 0.062103, 0.06166, 0.06166, 0.06166],
                    [19.715615, 24.855295, 23.742254, 23.961965, 23.961965, 23.961965] ]

network_state = [10**6 * 2, 10**6 , 10**6 * 3, 10**6, 10**6 * 0.9, 10**6 * 1.5]


input_num, input_chanel, height, width = original_tensor.size()
# 暂时没有计入 width， 先计算其余值
temp_flops = 2 * height *(in_chanel * kernerl_size * kernerl_size  + 1) * out_chanel
temp_num = input_num * in_chanel * height
# 主要是针对 width 进行划分，考虑计算时间和通信时间
com_k = computing_power[0]
com_b = computing_power[1]
com_k_back = []
for i in range(datanode_num):
    com_k_back.append(1.0 / com_k[i])
sum_com_k = sum(com_k_back[ 0 : datanode_num])

# widtgh 初始化
width_list = []
for it in range(datanode_num):
    if it == 0:
        width_list.append(int(sum(com_k_back[ 0 : it + 1]) / sum_com_k * width))
    else:
        width_list.append(
            int(sum(com_k_back[ 0: it + 1]) / sum_com_k * width) - int(sum(com_k_back[0 : it]) / sum_com_k * width))
# 通信时间和计算时间的定义和初始化
com_time = []
trans_time_forward = []
trans_time_backward = []
total_time = []
for it in range(datanode_num):
    # 初始化
    com_time.append(com_k[it] * (temp_flops * width_list[it] / 10 ** 6) + com_b[it])
    if it == 0 or it == datanode_num - 1:
        trans_time_forward.append((temp_num * (width_list[it] + cross_layer) * 4) / network_state[it] * (10 ** 3))
        trans_time_backward.append((temp_num * cross_layer * 4) / network_state[it] * (10 ** 3))
    else:
        trans_time_forward.append((temp_num * (width_list[it] + 2 * cross_layer) * 4) / network_state[it] * (10 ** 3))
        trans_time_backward.append((temp_num * cross_layer * 2 * 4) / network_state[it] * (10 ** 3))
    total_time.append(com_time[it] + trans_time_forward[it] + trans_time_backward[it])
# 定义 时间模型，T = a * x + b + c，a * x + b为计算时间，c为通信时间，
print (total_time)


# 开始迭代求最优情况
max_value = max(total_time)
min_value = min(total_time)
avg_value = np.mean(total_time)
last_diff = (max_value - min_value) * 2
while True:
    max_value = max(total_time)
    min_value = min(total_time)
    avg_value = np.mean(total_time)
    max_index = total_time.index(max_value)
    min_index = total_time.index(min_value)
    print("max_index: %d, min_index: %d" % (max_index, min_index))

    # max 和 min 满足结束条件
    diff = max_value - min_value
    print ("差值：%f" % (max_value - min_value))
    if diff <= 4:
        break
    else:
        step = int((max_value - avg_value)/com_k[max_index]/temp_flops * 10**6)
        width_list[max_index] -= step
        width_list[min_index] += step
        # 更新max_index对应的值
        if max_index == 0 or max_index == datanode_num - 1:
            total_time[max_index] = com_k[max_index] * (temp_flops * width_list[max_index] / 10**6) + trans_time[max_index]
        else:
            total_time[max_index] = com_k[max_index] * (temp_flops * width_list[max_index] / 10**6) + trans_time[max_index]
        # 更新min_index对应的值
        if min_index == 0 or min_index == datanode_num - 1:
            total_time[min_index] = com_k[min_index] * (temp_flops * width_list[min_index] / 10**6) + trans_time[min_index]
        else:
            total_time[min_index] = com_k[min_index] * (temp_flops * width_list[min_index] / 10**6) + trans_time[min_index]
        print (width_list)
        print("Total time :", total_time)
    print ("\n")
# 此处优化 width ，结束


# width 优化完成后，正式划分Tensor
start = 0
end = 0
divided_tensor = []
temp_tensor = torch.rand(1, 1, 1, 1)
print ("width_list:", width_list)
for it in range(datanode_num):
    end = start + width_list[it]
    if it == 0:
        # 最左边划分
        temp_tensor = original_tensor[:, :, :, start : int(end + cross_layer)]
    elif it == datanode_num - 1:
        # 最右边划分
        temp_tensor = original_tensor[:, :, :, int(start - cross_layer):end]
    else:
        # 中间非边界情况
        temp_tensor = original_tensor[:, :, :, int(start - cross_layer) : int(end + cross_layer)]
    # 循环变换下标
    start = end
    # 放入list
    divided_tensor.append(temp_tensor)

for it in range(datanode_num):
    print (divided_tensor[it].size())








# 得出一个tensor的bytes长度为 4， 不会改变
# def get_per_tensor_bytes_lenth(input_tensor):
#     input_num, input_chanel, height, width = input_tensor.size()
#     input_numpy = input.detach().numpy()
#     input_bytes = input_numpy.tobytes()
#     numbers = input_num * input_chanel * height * width
#     print ("input_bytes : %d" % len(input_bytes))
#     print ("Tensor numbers : %d" % numbers)
#     print ("Per tensor bytes length: ", len(input_bytes)/numbers )
#     print ("\n")
# def get_tensor_bytes_length( input_tensor ):
#     input_num, input_chanel, height, width = input_tensor.size()
#     numbers = input_num * input_chanel * height * width
#     bytes_length = int(numbers * 4)
#     return bytes_length
#
#
# input = torch.rand(1, 3, 224, 224)
# get_per_tensor_bytes_lenth(input)
# input = torch.rand(1, 3, 112, 112)
# get_per_tensor_bytes_lenth(input)
#
# input = torch.rand(1, 3, 224, 125)
# get_per_tensor_bytes_lenth(input)
# input = torch.rand(1, 3, 186, 112)
# get_per_tensor_bytes_lenth(input)
#
# get_tensor_bytes_length(input)









# def get_end_layer(start = 1, maxpool_layer = []):
#     max_value = max(maxpool_layer)
#     if start > max_value or start < 1:
#         return 0
#     for i in maxpool_layer:
#         if i > start:
#             return i
#
# maxpool_layer = [3, 6, 10, 14, 18]
# for i in range(1, 22, 1):
#     print (get_end_layer(i, maxpool_layer))


# ########## 测试 VGG16 的运行时间 ###############
# inference_model = VGG_model()
# # 循环计算VGG16网络,获得网络人工划分的长度
# conv_length = inference_model.get_conv_length()
# total_length = inference_model.get_total_length()
# # 输入参数大小
# input = torch.rand(1, 3, 224, 224)
# count = 10
# start_time = time.time()
# for i in range( count ):
#     output = inference_model(input, 1, total_length)
# end_time = time.time()
# print("Used time: %0.4fs" % ((end_time - start_time)/count))



# length = 102
# input_tensor = torch.rand(1, 64, 224, length)
# _, _, _, maxpool_length_1 = inference_model(input_tensor, 3, 3).size()
# print (maxpool_length_1)
#
#
# input_tensor = torch.rand(1, 64, 224, 224 - length)
# _, _, _, maxpool_length_2 = inference_model(input_tensor, 3, 3).size()
# print (maxpool_length_2)
#
#
# if 224/2 == maxpool_length_1 + maxpool_length_2:
#     print ("%d + %d = %d " %(maxpool_length_1, maxpool_length_2, 224/2))
# else:
#     print("%d + %d = %d < %d" % (maxpool_length_1, maxpool_length_2, maxpool_length_1 + maxpool_length_2, 224/2))
#
# tensor = torch.rand(1, 2, 3).size()
# print ("torch.rand(1): ", tensor )