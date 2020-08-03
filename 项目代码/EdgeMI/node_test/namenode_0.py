import sys
sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.num_set_up import Num_set_up
import torch
import threading
import time
import torch.nn as nn
from VGG.mydefine_VGG13 import VGG_model
from VGG.tensor_op import tensor_divide, tensor_divide_and_fill, tensor_divide_by_computing_and_fill, \
    tensor_divide_by_computing_network_and_fill, tensor_divide_by_computing_and_network
from VGG.tensor_op import merge_total_tensor, merge_part_tensor
from other.network_and_computing_record import Network_And_Computing

# 适用于一下 四种 不同的场景
# 1、各设备等算力、通信资源相同，全部计算数据交换
# 2、各设备算力描述不同、通信资源相同，单层、差异数据交换， 池化层再次全部交换
# 3、各设备算力描述不同、通信资源不同，单层、差异数据交换， 池化层非全部全部交换
# 4、各设备算力描述不同、通信资源不同，多层、差异数据交换， 池化层再次全部交换

# 人工设定参数
num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()
# 获得 datanode 设备的 计算能力 和 通信能力
network_and_computing = Network_And_Computing()
computing_power = network_and_computing.get_computing_power(datanode_num)
network_state = network_and_computing.get_network_state(datanode_num)
computing_a = network_and_computing.get_computing_a(datanode_num)
computing_b = network_and_computing.get_computing_b(datanode_num)

# 初始化，加载网络
inference_model = VGG_model()
# 循环计算VGG16网络,获得网络人工划分的长度
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()
c_out_list = inference_model.get_c_out()
# 池化层记录
maxpool_layer = inference_model.get_maxpool_layer()
# 模拟VGG16网络输入
width = 448
input = torch.rand(1, 3, width, width)
# 全局变量，保存接收到的中间结果
temp_receive_tensor = input
# 通信网络初始化
namenode = Network_init_namenode(namenode_num = namenode_num, datanode_num = datanode_num)
# 创建线程list
thread = []
recv_tensor_list = []
for j in range(datanode_num):
    recv_tensor_list.append(0)
    thread.append(0)

def send_total_data(datanode_name, input_tensor ,start, end):
    # 发送 total 数据
    namenode.namenode_send_data(datanode_name = datanode_name, input_tensor = input_tensor, start = start, end = end )
    # 成功发送并接受计算结果，主线程等待datanode完成
def send_part_data(datanode_name, input_tensor, start, end):
    # 发送 part 数据
    namenode.namenode_send_data(datanode_name = datanode_name, input_tensor = input_tensor, start = start, end = end)
    # 成功发送并接受计算结果，主线程等待datanode完成
def get_end_layer(start = 1, maxpool_layer = []):
    max_value = max(maxpool_layer)
    if start > max_value or start < 1:
        return 0
    for layer in maxpool_layer:
        if layer > start:
            return layer


# 后期根据计算能力， 通信能力，多设备划分
# 一开始，测出回归模型：

if __name__ == "__main__":


    # # 11111111111111111111111111111111111111111111
    # # 此刻运行：（各设备等算力、全部计算数据交换）
    # # 进入到循环计算、通信中
    # time.sleep(3)
    # middle_output = input
    # # 循环按照神经网络的结构
    # start_time = time.time()
    # for layer_it in range(1, conv_length + 1, 1):
    #     # 暂定全连接层， 池化层在namenode运行
    #     if layer_it == conv_length: # 最后一层
    #         final_output = inference_model(middle_output, layer_it, layer_it)
    #     elif layer_it in maxpool_layer:
    #         middle_output = inference_model(middle_output, layer_it, layer_it)
    #     else:
    #         # 普通的卷积层有datanode共同运行，此刻运行：（各设备等算力、全部计算数据交换）
    #         divided_tensor, divide_record = tensor_divide_by_computing_and_fill(middle_output,
    #                                                                             datanode_num = datanode_num,
    #                                                                             cross_layer = 1,
    #                                                                             computing_power = computing_power)
    #         # 设置线程将divided_tensor依次发送给各个datanode，并等待返回计算结果（），将计算值放入大矩阵之中暂存
    #         # 难题：中间计算结果如何保存才能被主线程获取
    #         for i in range(datanode_num):
    #             # 创建线程发送，线程参数包括：datanode编号， 计算数据（start， end， send_tensor）, 返回
    #             # print ("sned:", i)
    #             thread[i] = threading.Thread(target = send_total_data, args=(namenode, i, layer_it, layer_it, divided_tensor[i]))
    #             thread[i].start()
    #         # 等待所有线程完成
    #         for i in range(datanode_num):
    #             thread[i].join()
    #         # 整合之后的Tensor
    #         merged_tensor = namenode.get_merged_total_tensor()
    #         # 结束本layer的推理，继续推理下一个layer。构造一个新的输入
    #         middle_output = merged_tensor
    #     print ("middle_output: ", middle_output.size())
    #     print ("结束 %d的推理" % layer_it)
    # end_time = time.time()
    # print("Used time: %0.3fs" % (end_time - start_time))
    # # 关闭连接
    # time.sleep(2)
    # namenode.close_all()
    # print("关闭 NameNode 所有 Socket 连接")

    # # 2222222222222222222222222222222222222222222222222222222222
    # # 此刻运行：（各设备算力描述不同、通信资源相同，单层、差异数据交换） 池化层再次全部交换
    # time.sleep(1)
    # middle_output = input
    # final_output = torch.rand(1, 100)
    # start_time = time.time()
    # for layer_it in range(1, conv_length + 1, 1):
    #     print ("计算第 %d 层" % layer_it)
    #     # “全连接层” 和 “池化层” 运行在 namenode，“全连接层”运行在datanode，全连接层的多机计算创新 暂时不做
    #     if layer_it == conv_length:
    #         # 已计算到卷积层最后一层，后续由 namenode 计算，“全连接层”计算 暂时不做
    #         final_output = inference_model(middle_output, layer_it, total_length)
    #         print("计算全连接层")
    #     # 若为 池化层, namenode 直接计算
    #     elif layer_it in maxpool_layer:
    #         # middle_output = namenode.get_merged_total_tensor()
    #         middle_output = inference_model(middle_output, layer_it, layer_it)
    #         print ("池化层输出数据大小：", middle_output.size())
    #
    #     # “第一层”或者“池化层” 后的一层，需要将整个特征图按照 datanode_num及其他条件 划分后的数据全部传输
    #     elif (layer_it == 1) or (layer_it - 1 in maxpool_layer):
    #         print ("第一层 或者 池化层 后一层输出")
    #         divided_tensor_list, divide_record = tensor_divide_by_computing_and_fill(middle_output, datanode_num = datanode_num, cross_layer = 1,  computing_power = computing_power)
    #         for i in range(datanode_num):
    #             # 创建线程发送，线程参数包括：datanode编号， 计算数据（send_tensor, start，end）, 返回
    #             # print ("sned:", i)
    #             thread[i] = threading.Thread(target = send_total_data, args = (i, divided_tensor_list[i], layer_it, layer_it))
    #             thread[i].start()
    #         # 等待所有线程完成
    #         for i in range(datanode_num):
    #             thread[i].join()
    #         # 部分数据 整合之后的Tensor，为list格式
    #             recv_tensor_list = namenode.get_recv_tensor_list()
    #     # 非上述三种种方法，普通的“卷积层”计算, 初步将将上一 layer 接收的 list 作为数据再次发送回去
    #     else:
    #         # 设置线程将divided_tensor依次发送给各个datanode，并等待返回计算结果
    #         divided_tensor = recv_tensor_list
    #         for i in range(datanode_num):
    #             # 创建线程发送，线程参数包括：datanode编号， 计算数据（start， end， send_tensor）, 返回
    #             thread[i] = threading.Thread(target = send_part_data, args=(i, divided_tensor[i], layer_it, layer_it))
    #             thread[i].start()
    #         # 等待所有线程完成
    #         for i in range(datanode_num):
    #             thread[i].join()
    #         # 接受的 tensor 合并
    #         recv_tensor_list = namenode.get_recv_tensor_list()
    #         # 如果下一层是 池化层，将数据合并作为middle_output
    #         ################ 当前的代码重点 ################
    #         if layer_it + 1 in maxpool_layer:
    #             # 与datanode的内部数据有关
    #             middle_output = namenode.get_merged_total_tensor()
    # # 结束全部 “”和“”的计算
    # print ("NameNode 结束计算")
    # print ("Final output: ", final_output.size())
    # # 打印相关统计结果
    # end_time = time.time()
    # print("Used time: %0.3fs" % (end_time - start_time))
    # # 关闭连接
    # time.sleep(2)
    # namenode.close_all()
    # print("关闭 NameNode 所有 Socket 连接")

    # 以上部分代码完成


    # # 3333333333333333333333333333333333333333333333333333333333333333333
    # # 此刻运行：各设备算力描述不同、通信资源不同，单层、差异数据交换， 池化层非全部全部交换
    # time.sleep(1)
    # middle_output = input
    # final_output = torch.rand(1, 100)
    # start_time = time.time()
    # for layer_it in range(1, conv_length + 1, 1):
    #     print("计算第 %d 层" % layer_it)
    #     # “全连接层” 和 “池化层” 运行在 namenode，“全连接层”运行在datanode，全连接层的多机计算创新 暂时不做
    #     if layer_it == conv_length:
    #         # 已计算到卷积层最后一层，后续由 namenode 计算，“全连接层”计算 暂时不做
    #         final_output = inference_model(middle_output, layer_it, total_length)
    #         print("计算全连接层")
    #     # 若为 池化层, namenode 直接计算， 该部分如何修改？ #######################################################
    #     elif layer_it in maxpool_layer:
    #         # middle_output = namenode.get_merged_total_tensor()
    #         middle_output = inference_model(middle_output, layer_it, layer_it)
    #         print("池化层输出数据大小：", middle_output.size())
    #
    #     # “第一层”或者“池化层” 后的一层，需要将整个特征图按照 datanode_num及其他条件 划分后的数据全部传输
    #     elif (layer_it == 1) or (layer_it - 1 in maxpool_layer):
    #         print("第一层 或者 池化层 后一层输出")
    #         divided_tensor_list, divide_record = tensor_divide_by_computing_and_fill(middle_output,
    #                                                                                  datanode_num = datanode_num,
    #                                                                                  cross_layer=1,
    #                                                                                  computing_power=computing_power)
    #         for i in range(datanode_num):
    #             # 创建线程发送，线程参数包括：datanode编号， 计算数据（send_tensor, start，end）, 返回
    #             # print ("sned:", i)
    #             thread[i] = threading.Thread(target=send_total_data,
    #                                          args=(i, divided_tensor_list[i], layer_it, layer_it))
    #             thread[i].start()
    #         # 等待所有线程完成
    #         for i in range(datanode_num):
    #             thread[i].join()
    #             # 部分数据 整合之后的Tensor，为list格式
    #             recv_tensor_list = namenode.get_recv_tensor_list()
    #     # 非上述三种种方法，普通的“卷积层”计算, 初步将将上一 layer 接收的 list 作为数据再次发送回去
    #     else:
    #         # 设置线程将divided_tensor依次发送给各个datanode，并等待返回计算结果
    #         divided_tensor = recv_tensor_list
    #         for i in range(datanode_num):
    #             # 创建线程发送，线程参数包括：datanode编号， 计算数据（start， end， send_tensor）, 返回
    #             thread[i] = threading.Thread(target=send_part_data, args=(i, divided_tensor[i], layer_it, layer_it))
    #             thread[i].start()
    #         # 等待所有线程完成
    #         for i in range(datanode_num):
    #             thread[i].join()
    #         # 接受的 tensor 合并
    #         recv_tensor_list = namenode.get_recv_tensor_list()
    #         # 如果下一层是 池化层，将数据合并作为middle_output
    #         ################ 当前的代码重点 ################
    #         if layer_it + 1 in maxpool_layer:
    #             # 与datanode的内部数据有关
    #             middle_output = namenode.get_merged_total_tensor()
    # # 结束全部 “”和“”的计算
    # print("NameNode 结束计算")
    # print("Final output: ", final_output.size())
    # # 打印相关统计结果
    # end_time = time.time()
    # print("Used time: %0.3fs" % (end_time - start_time))
    # # 关闭连接
    # time.sleep(2)
    # namenode.close_all()
    # print("关闭 NameNode 所有 Socket 连接")

    # 444444444444444444444444444444444444444444444444444444444444444444444444
    # 此刻运行：各设备算力描述不同、通信资源不同，多层、差异数据交换， 池化层再次全部交换
    time.sleep(1)
    middle_output = input
    # 将第一层加入，便于整体计算，以 maxpool_layer 为界限
    final_output = torch.rand(1, 100)
    start_time = time.time()
    if datanode_num != 1:
        for layer_it in range(1, conv_length + 1, 1):
            # “全连接层” 和 “池化层” 运行在 namenode，“全连接层”运行在datanode，全连接层的多机计算创新 暂时不做
            if layer_it == conv_length:
                # 已计算到卷积层最后一层，后续由 namenode 计算，“全连接层”计算 暂时不做
                final_output = inference_model(middle_output, layer_it, total_length)
                print("计算全连接层")
            # 若为 池化层, namenode 直接计算，该部分如何修改？ #######################################################
            elif layer_it in maxpool_layer:
                # 该部分是卷积层运算
                middle_output = inference_model(middle_output, layer_it, layer_it)
                print("池化层输出数据大小：", middle_output.size())

            # “第一层”或者“池化层” 后的一层，需要将整个特征图按照 datanode_num及其他条件 划分后的数据全部传输（该部分存在问题）
            elif (layer_it == 1) or (layer_it - 1 in maxpool_layer):
                # 此处的难点是：如何确定 start - end ，(start 与 end 写法没有问题)
                start = layer_it
                end = get_end_layer(start, maxpool_layer) - 1
                cross_layer = end - start + 1
                print("cross_layer: %d" % cross_layer)
                # 此处更换 划分函数
                # divided_tensor_list, divide_record = tensor_divide_by_computing_and_fill(middle_output,
                #                                                                          datanode_num = datanode_num,
                #                                                                          cross_layer = cross_layer,
                #                                                                          computing_power = computing_power)
                divided_tensor_list, divide_record = tensor_divide_by_computing_and_network(middle_output, datanode_num = datanode_num,
                                                                                            cross_layer = cross_layer, computing_power = computing_power,
                                                                                            computing_a = computing_a, computing_b = computing_b,
                                                                                            network_state = network_state,c_out = c_out_list[layer_it])

                for i in range(datanode_num):
                    # 创建线程发送，线程参数包括：datanode编号， 计算数据（send_tensor, start，end）, 返回
                    # print ("sned:", i)
                    thread[i] = threading.Thread(target = send_total_data, args = (i, divided_tensor_list[i], start, end))
                    thread[i].start()
                # 等待所有线程完成
                for i in range(datanode_num):
                    thread[i].join()
                # 部分数据 整合之后的Tensor，
                temp = namenode.get_recv_tensor_list()
                middle_output = namenode.get_merged_total_tensor(cross_layer = cross_layer)
                print ("合并后的 middle_output：", middle_output.size())
            # 非上述三种种方法，普通的“卷积层”计算，不要这种情况，直接终止本次计算
            else:
                print ("NameNode不参与第 %d 层计算" % layer_it)
                continue
    else:
        for layer_it in range(1, conv_length + 1, 1):
            # “全连接层” 和 “池化层” 运行在 namenode，“全连接层”运行在datanode，全连接层的多机计算创新 暂时不做
            if layer_it == conv_length:
                # 已计算到卷积层最后一层，后续由 namenode 计算，“全连接层”计算 暂时不做
                final_output = inference_model(middle_output, layer_it, total_length)
                print("计算全连接层")
            # 若为 池化层, namenode 直接计算，该部分如何修改？ #######################################################
            elif layer_it in maxpool_layer:
                # 该部分是卷积层运算
                middle_output = inference_model(middle_output, layer_it, layer_it)
                print("池化层输出数据大小：", middle_output.size())

            # “第一层”或者“池化层” 后的一层，需要将整个特征图按照 datanode_num及其他条件 划分后的数据全部传输（该部分存在问题）
            elif (layer_it == 1) or (layer_it - 1 in maxpool_layer):
                # 此处的难点是：如何确定 start - end ，(start 与 end 写法没有问题)
                start = layer_it
                end = get_end_layer(start, maxpool_layer) - 1
                cross_layer = end - start + 1
                print("cross_layer: %d" % cross_layer)

                for i in range(datanode_num):
                    # 创建线程发送，线程参数包括：datanode编号， 计算数据（send_tensor, start，end）, 返回
                    # print ("sned:", i)
                    thread[i] = threading.Thread(target=send_total_data, args=(i, middle_output, start, end))
                    thread[i].start()
                # 等待所有线程完成
                for i in range(datanode_num):
                    thread[i].join()
                # 部分数据 整合之后的Tensor
                temp = namenode.get_recv_tensor_list()
                middle_output = namenode.get_merged_total_tensor(cross_layer = cross_layer)
                print ("合并后的 middle_output：", middle_output.size())
            # 非上述三种种方法，普通的“卷积层”计算，不要这种情况，直接终止本次计算
            else:
                print ("NameNode不参与第 %d 层计算" % layer_it)
                continue

    # 结束全部 “”和“”的计算
    print("NameNode 结束计算")
    print("Final Output: ", final_output.size())
    # 打印相关统计结果
    end_time = time.time()
    print("Used Time: %0.3fs" % (end_time - start_time))
    # 关闭连接
    time.sleep(2)
    namenode.close_all()
    print("关闭 NameNode 所有 Socket 连接")


