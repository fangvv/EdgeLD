import sys
sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.num_set_up import Num_set_up
from VGG.mydefine_VGG13 import VGG_model
from VGG.tensor_op import divied_middle_output
import torch.nn as nn
import torch, time
import threading

# 初始参数设置
num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()
datanode_name = 4
cross_layer = 1

# 加载、初始化模型
inference_model = VGG_model()
# 初始化网络通信模型，参数人为设定
datanode = Network_init_datanode(namenode_num = namenode_num, datanode_num = datanode_num, datanode_name = datanode_name)
# 获得模型中人工设定的参数
maxpool_layer = inference_model.get_maxpool_layer()
# 循环计算VGG16网络,获得网络人工划分的长度
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()
# 适用于一下 四种 不同的场景
# 1、各设备等算力、通信资源相同，全部计算数据交换
# 2、各设备算力描述不同、通信资源相同，单层、差异数据交换， 池化层再次全部交换
# 3、各设备算力描述不同、通信资源不同，单层、差异数据交换， 池化层非全部全部交换
# 4、各设备算力描述不同、通信资源不同，多层、差异数据交换， 池化层再次全部交换



if __name__ == "__main__":

    # # 1111111111111111111111111111111111111111111111111111111111111111111111111111111
    # # 场景1： 各设备等算力、通信资源相同，全部计算数据交换
    # # datanode 接受来自 namenode 的全部数据，直接进行推理，无需考虑数据的缓存与划分
    # # 连续接收 namenode 发来的计算请求，直接计算并且返回推理结果
    # print("进入计算场景 1")
    # while True:
    #     print("#### 进入模型阶段推理 ####")
    #     # 接受来自 namenode 发送的参数
    #     start, end, recv_tensor = datanode.datanode_recv_data()
    #     middle_output = inference_model(recv_tensor, start, end)
    #     # print ("middle_output:", middle_output.size())
    #     datanode.datanode_send_data(middle_output, start, end)
    #     print ("完成 %d - %d 的推理任务，并返回推理结果" %(start, end) )
    #     if end >= conv_length - 1:
    #         print("DataNode %d 结束推理")
    #         break
    #     # break
    # # 关闭已建立的连接
    # time.sleep(2)
    # datanode.close()
    # print("关闭 DataNode %d 的Socket连接" % datanode_name)

    # 已全部完成


    # # 2222222222222222222222222222222222222222222222222222222222222222222222222222222
    # # 场景2： 各设备算力描述不同、通信资源相同，单层、差异数据交换， 池化层再次全部交换
    # # datanode 接受来自 namenode 的 全部数据 或者 部分差分数据，分情况进行推理，需要考虑数据的缓存与划分
    # # 连续接收 namenode 发来的计算请求，分情况 计算并且返回推理结果
    # print("进入计算场景 2")
    # print("#### 进入模型阶段推理 ####\n")
    # while True:
    #     # 接受来自 namenode 发送的参数
    #     start, end, recv_tensor = datanode.datanode_recv_data()
    #     print ("接收来自 NameNode 的数据 recv_tensor：", recv_tensor.size())
    #     print ("要求计算 %d - %d " %(start, end))
    #     print ("datanode.get_last_inference_layer() : ", datanode.get_last_inference_layer())
    #
    #     # 是否是第一层推理( 通过判断 datanode 设定的初始化 ) 或者是 池化层的后一层。需要接受可直接计算的 tensor
    #     if ( datanode.get_last_inference_layer() == 0 ) or ( start - 1 in maxpool_layer ): # 当前判断条件还不确定
    #         # recv_tensor为完整的计算tensor，不需要合并或者拆分
    #         print ("1111111111111111111111111111111111111111111111111")
    #         middle_output = inference_model(recv_tensor, start, end)
    #         print ("计算第 %d 层的 middle_output:" % start , middle_output.size())
    #
    #         # 如果下一层是 maxpool， 需要将数据全部发送至 namenode
    #         if end + 1 in maxpool_layer:
    #             print("进入清空代码")
    #             datanode.datanode_send_data(middle_output, start, end)
    #             # 清空暂存的数据
    #             datanode.empty_tensor()
    #             print("发送第 %d 层的 send_tensor:" % start, middle_output.size())
    #         else:
    #             saved_tensor, divied_tensor_list = divied_middle_output(input_tensor=middle_output,
    #                                                                     datanode_num=datanode_num,
    #                                                                     datanode_name=datanode_name,
    #                                                                     cross_layer=1)
    #             # 计算结果拆分并保存至 datanode 中
    #             datanode.set_saved_tensor(saved_tensor)
    #             datanode.set_divied_tensor_list(divied_tensor_list)
    #             # 将 divied_tensor 合并并作为发送数据
    #             send_tensor = datanode.get_divied_merged_tensor()
    #             print("发送第 %d 层的 send_tensor:" % start, send_tensor.size())
    #             # 发送数据至 namenode
    #             datanode.datanode_send_data(send_tensor, start, end)
    #         # 设置记录参数
    #         datanode.set_last_inference_layer(end)
    #     # 非上述情况，之前的推理过程已有数据保存，将 部分数据 和 已接收数据 合并为新的 tensor 同于计算。
    #     else:
    #         # 对于接收的参数不做处理，处理之前保存的数据saved_tensor、divied_tensor 合并
    #         print(" 2222222222222222222222222222222222222222222222222 ")
    #         merged_tensor = datanode.get_merged_tensor()
    #         middle_output = inference_model(merged_tensor, start, end)
    #         print("计算第 %d 层的 middle_output:" % start, middle_output.size())
    #
    #         # 如果下一层是 maxpool， 需要将数据全部发送至 namenode
    #         if end + 1 in maxpool_layer:
    #             print ("进入清空代码")
    #             datanode.datanode_send_data(middle_output, start, end)
    #             # 清空暂存的数据
    #             datanode.empty_tensor()
    #             print("发送第 %d 层的 send_tensor:" % start, middle_output.size())
    #         else:
    #             saved_tensor, divied_tensor = divied_middle_output(input_tensor=middle_output,
    #                                                                datanode_num=datanode_num,
    #                                                                datanode_name=datanode_name,
    #                                                                cross_layer = 1)
    #             # 计算结果拆分并保存至 datanode 中
    #             datanode.set_saved_tensor(saved_tensor)
    #             datanode.set_divied_tensor_list(divied_tensor)
    #             # 将 divied_tensor 合并并作为发送数据
    #             send_tensor = datanode.get_divied_merged_tensor()
    #             print("发送第 %d 层的 send_tensor:" % start, send_tensor.size())
    #             datanode.datanode_send_data(send_tensor, start, end)
    #             # 设置记录参数
    #         datanode.set_last_inference_layer(end)
    #     print("完成%d - %d 的推理任务，并返回计算结果\n" % (start, end))
    #     if end >= conv_length - 1:
    #         print("DataNode %d 结束推理")
    #         break
    # # 关闭已建立的连接
    # time.sleep(2)
    # datanode.close()
    # print("关闭 DataNode %d 的Socket连接" % datanode_name)





    # # 333333333333333333333333333333333333333333333333333333333333333333333333333333333
    # # 场景3： 各设备算力描述不同、通信资源不同，单层、差异数据交换， 池化层再次全部交换
    # # datanode 接受来自 namenode 的 全部数据 或者 部分差分数据，分情况进行推理，需要考虑数据的缓存与划分
    # # 连续接收 namenode 发来的计算请求，分情况 计算并且返回推理结果
    # print("进入计算场景 3")
    # while True:
    #     print("#### 进入模型阶段推理 ####")
    #     # 接受来自 namenode 发送的参数
    #     start, end, recv_tensor = datanode.datanode_recv_data()
    #
    #     # 是否是第一层推理( 通过判断datanode设定的初始化 ) 或者是 池化层的后一层。需要接受可直接计算的 tensor
    #     if (datanode.get_saved_tensor() == 0) or (
    #             datanode.get_current_inference_layer + 1 in maxpool_layer):  # 当前判断条件还不确定
    #         # start, end, recv_tensor = datanode.datanode_recv_data()
    #         middle_output = inference_model(recv_tensor, start, end)
    #         # print ("middle_output:", middle_output.size())
    #         saved_tensor, divied_tensor = divied_middle_output(input_tensor=middle_output,
    #                                                            datanode_num=datanode_num,
    #                                                            hostname=hostname,
    #                                                            cross_layer=1)
    #         # 数据保存至datanode中
    #         datanode.set_saved_tensor(saved_tensor)
    #         datanode.set_divied_tensor(divied_tensor)
    #         # 将divied_tensor合并为一个并发送
    #         send_tensor = datanode.get_divied_merged_tensor()
    #         datanode.datanode_send_data(middle_output, start, end)
    #         print("完成%d - %d 的推理任务，并返回推理结果" % (start, end))
    #     # 非上述情况，之前的推理过程已有数据保存，将 部分数据 和 已接收数据 合并为新的 tensor 同于计算。
    #     else:
    #         # start, end, recv_tensor = datanode.datanode_recv_data()
    #         # 得到合成后的数据
    #         merged_tensor = datanode.get_part_merged_tensor()
    #         middle_output = inference_model(merged_tensor, start, end)
    #         # print ("middle_output:", middle_output.size())
    #         saved_tensor, divied_tensor = divied_middle_output(input_tensor=middle_output,
    #                                                            datanode_num=datanode_num,
    #                                                            hostname=hostname,
    #                                                            cross_layer=1)
    #         # 数据保存至datanode中
    #         datanode.set_saved_tensor(saved_tensor)
    #         datanode.set_divied_tensor(divied_tensor)
    #         # 将divied_tensor合并为一个并发送
    #         send_tensor = datanode.get_divied_merged_tensor()
    #         datanode.datanode_send_data(middle_output, start, end)
    #         print("完成%d - %d 的推理任务，并返回推理结果" % (start, end))
    #         # 如果下一层为maxpool layer，需要清空部分数据
    #     if end >= maxpool_layer[-1] - 1:
    #         print("结束推理")
    #         break
    # # 关闭已建立的连接
    # time.sleep(5)
    # datanode.close()
    # print("关闭 DataNode %d 的Socket连接" % hostname)


    # 4444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
    # 场景4： 各设备算力描述不同、通信资源不同，多层、差异数据交换， 池化层再次全部交换
    # datanode 接受来自 namenode 的 全部数据 或者 部分差分数据，分情况进行推理，需要考虑数据的缓存与划分
    # 连续接收 namenode 发来的计算请求，分情况 计算并且返回推理结果
    print("进入计算场景 4")
    print("#### 进入模型阶段推理 ####\n")
    while True:
        # 接受来自 namenode 发送的参数
        start, end, recv_tensor = datanode.datanode_recv_data()
        print ("接收来自 NameNode 的数据 recv_tensor：", recv_tensor.size())
        print ("要求计算 %d - %d " % (start, end) )
        print ("datanode.get_last_inference_layer() : ", datanode.get_last_inference_layer())

        # 是否是第一层推理( 通过判断 datanode 设定的初始化 ) 或者是 池化层的后一层。需要接受可直接计算的 tensor
        if ( datanode.get_last_inference_layer() == 0 ) or ( start - 1 in maxpool_layer ): # 当前判断条件还不确定
            # recv_tensor为完整的计算tensor，不需要合并或者拆分
            middle_output = inference_model(recv_tensor, start, end)
            print ("计算第 %d 层的 middle_output:" % start , middle_output.size())
            # 如果下一层是 maxpool， 需要将数据全部发送至 namenode
            if end + 1 in maxpool_layer:
                datanode.datanode_send_data(middle_output, start, end)
                print("发送第 %d 层的 send_tensor:" % end, middle_output.size())
            # 设置 计算记录 参数
            datanode.set_last_inference_layer(end)
        # 仅有上述一种情况，其他情况暂时不讨论

        print("完成 %d - %d 的推理任务，并返回计算结果\n" % (start, end))
        if end >= conv_length - 1:
            print("DataNode %d 结束推理")
            break
    # 关闭已建立的连接
    time.sleep(2)
    datanode.close()
    print("关闭 DataNode %d 的Socket连接" % datanode_name)

