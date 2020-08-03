import sys
sys.path.append("../..")
sys.path.append("..")

import torch, time, socket, json, six
import torch.nn as nn
import numpy as np
from VGG.tensor_op import merge_total_tensor, merge_part_tensor


# IP设置
# namenode_ip = "127.0.0.1"
namenode_ip = "192.168.10.139"
# datanode_ip = ["127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"]
datanode_ip = ["192.168.10.140", "192.168.10.141", "192.168.10.142", "192.168.10.143", "192.168.10.144", "192.168.10.145"]
datanode_port = [10000, 10001, 10002, 10003, 10004, 10005]

class Network_init_namenode():
    def __init__(self, namenode_num = 1, datanode_num = 1):
        super(Network_init_namenode, self).__init__()
        print ("NameNode 开始初始化")
        self.datanode_num = datanode_num
        self.client_socket = []
        if (datanode_num >= 1):
            for hostname in range(datanode_num):
                tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                tcp_client_socket.connect((datanode_ip[hostname], datanode_port[hostname]))
                hello_world = "Hello DataNode "+ str(hostname) + ", I'm NameNode"
                tcp_client_socket.send(hello_world.encode())

                recv_data_test = tcp_client_socket.recv(1024)
                print (str(recv_data_test, encoding="UTF-8"))
                self.client_socket.append(tcp_client_socket)
        print ("NameNode 初始化完成")
        self.recv_tensor_temp_list = []
        for it in range(datanode_num):
            self.recv_tensor_temp_list.append(0)

    def get_recv_tensor_list(self):
        return self.recv_tensor_temp_list

    def get_merged_total_tensor(self, divide_record = 0, cross_layer = 1):
        temp = merge_total_tensor(self.recv_tensor_temp_list, divide_record = divide_record, cross_layer = cross_layer)
        # # 原先的recv_tensor_temp_list重新初始化
        # for i in range(self.datanode_num):
        #     self.recv_tensor_temp_list[i] = torch.rand(1, 1, 1, 1)
        return temp

    def get_merged_part_tensor(self):
        temp = merge_part_tensor(self.recv_tensor_temp_list, divide_record=0, cross_layer=1)
        # # 原先的recv_tensor_temp_list重新初始化
        # for i in range(self.datanode_num):
        #     self.recv_tensor_temp_list[i] = torch.rand(1, 1, 1, 1)
        return temp

    def namenode_send_data(self, datanode_name, input_tensor, start, end):
        # 先发送数据长度，再发送数据
        input_numpy = input_tensor.detach().numpy()
        start = str(start).encode(encoding='utf-8')
        end = str(end).encode(encoding='utf-8')
        input_numpy_size = get_numpy_size(input_tensor)
        input_bytes = input_numpy.tobytes()
        send_data = start + b'@#$%' + end + b'@#$%' + input_numpy_size + b'@#$%' + input_bytes
        # 发送长度
        send_data_len = str(len(send_data)).encode(encoding='utf-8')
        # print("send_data长度：", len(send_data))
        self.client_socket[datanode_name].send(send_data_len)
        time.sleep(0.01)
        # 发送数据
        self.client_socket[datanode_name].sendall(send_data)
        # print("namenode socket 数据发送完成")
        # print("send_return_info: ", send_return_info)

        # 发送数据后等待接收datanode返回的数据
        data_total_len = self.client_socket[datanode_name].recv(1024)
        data_total_len = int(str(data_total_len, encoding="UTF-8"))
        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.client_socket[datanode_name].recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp
        # print("最终接受数据的长度：", len(recv_data), recv_data_len)

        split_list = recv_data.split(b'@#$%')
        recv_start = int(str(split_list[0], encoding="UTF-8"))
        recv_end = int(str(split_list[1], encoding="UTF-8"))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        # print("recv_numpy_size: ", recv_numpy_size)
        recv_numpy = np.frombuffer(split_list[3], dtype = np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape = recv_numpy_size)
        recv_tensor = torch.from_numpy(recv_numpy)
        self.recv_tensor_temp_list[datanode_name] = recv_tensor
        return recv_start, recv_end, recv_tensor

    def namenode_recv_data(self, datanode_name):
        # 先接收数据长度，再接收数据
        # 接收的数据：start 0，end 0， numpy_size, tensor
        data_total_len = self.client_socket[datanode_name].recv(1024)
        data_total_len = int(str(data_total_len, encoding='utf-8'))
        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.client_socket[datanode_name].recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp
        # print("最终接受数据的长度：", len(recv_data), recv_data_len)

        split_list = recv_data.split(b'@#$%')
        start = int(str(split_list[0], encoding="UTF-8"))
        end = int(str(split_list[1], encoding="UTF-8"))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        # print("recv_numpy_size: ", recv_numpy_size)
        recv_numpy = np.frombuffer(split_list[3], dtype=np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
        recv_tensor = torch.from_numpy(recv_numpy)
        return start, end, recv_tensor

    def close(self, datanode_name):
        self.client_socket[datanode_name].close()
    def close_all(self):
        for i in range(self.datanode_num):
            self.client_socket[i].close()

class Network_init_datanode():
    def __init__(self, namenode_num = 1, datanode_num = 3, datanode_name = 0):
        super(Network_init_datanode, self).__init__()
        print ("DataNode %d 开始初始化" % datanode_name )
        # 创建服务器 socket
        tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_server_socket.bind((datanode_ip[datanode_name], datanode_port[datanode_name]))
        tcp_server_socket.listen(2)
        # 得到client socket
        self.datanode_socket, client_addr = tcp_server_socket.accept()
        recv_data_test = self.datanode_socket.recv(1024)
        # 简单的测试
        print (str(recv_data_test, encoding="UTF-8"))
        # 发送数据测试
        hello_world = "Hello NameNode, I have received your hello world, I'm DateNode " + str(datanode_name)
        self.datanode_socket.send(hello_world.encode())
        print("DataNode %d 初始化完成\n" % datanode_name)

        # 初始化一些参数，中间计算结果分割为saved_tensor和divied_tensor，saved_tensor保存在本机，divied_tensor发送至namenode
        self.datanode_num = datanode_num
        self.datanode_name = datanode_name
        self.last_inference_layer = 0
        self.saved_tensor = torch.rand(1, 1, 1, 1) # tensor数据保存在本机
        self.divied_tensor_list = [] # 数据格式为list，发送至namenode
        # 最左侧 或 最右侧, 数据初始化
        if datanode_name==0 or datanode_name == datanode_num-1:
            self.divied_tensor_list.append(torch.rand(1, 1, 1, 1))
        # 中间的数据，左右两侧都会拆除，数据初始化
        else:
            self.divied_tensor_list.append(torch.rand(1, 1, 1, 1))
            self.divied_tensor_list.append(torch.rand(1, 1, 1, 1))
    # set, get 函数
    def set_last_inference_layer(self, layer):
        self.last_inference_layer = layer
    def set_saved_tensor(self, tensor):
        self.saved_tensor = tensor
    def set_divied_tensor_list(self, tensor_list):
        self.divied_tensor_list = tensor_list
    def get_last_inference_layer(self):
        return int(self.last_inference_layer)
    def get_saved_tensor(self):
        return self.saved_tensor
    # 根据 datanode_name 的不同，将一个或两个divied_tensor
    def get_divied_merged_tensor(self):
        if self.datanode_name == 0 or self.datanode_num - 1:
            return self.divied_tensor_list[0]
        else:
            return torch.cat((self.divied_tensor_list[0], self.divied_tensor_list[1]), 3)
    # 根据 datanode_name 的不同，合并 saved_tensor 和 divied_tensor_list
    def get_merged_tensor(self):

        if self.datanode_name == 0:
            merged_tensor = torch.cat((self.saved_tensor, self.divied_tensor_list[0]), 3)
        elif self.datanode_name == self.datanode_num - 1:
            merged_tensor = torch.cat((self.divied_tensor_list[0], self.saved_tensor), 3)
        else:
            merged_tensor = torch.cat((self.divied_tensor_list[0], self.saved_tensor, self.divied_tensor_list[1]), 3)
        return merged_tensor
    # 根据要求清空saved_tensor 和 divied_tensor_list
    def empty_tensor(self):
        self.saved_tensor = torch.rand(1, 1, 1, 1)
        if self.datanode_name==0 or self.datanode_name == self.datanode_num-1:
            self.divied_tensor_list[0] = torch.rand(1, 1, 1, 1)
        # 中间的数据，左右两侧都会拆除，数据初始化
        else:
            self.divied_tensor_list[0] = torch.rand(1, 1, 1, 1)
            self.divied_tensor_list[1] = torch.rand(1, 1, 1, 1)


    def datanode_send_data(self, input_tensor, start=0, end=0):
        # 先发送数据长度，再发送数据
        # print ("datanode_socket 数据发送开始")
        input_numpy = input_tensor.detach().numpy()

        start = str(start).encode(encoding="UTF-8")
        end = str(end).encode(encoding="UTF-8")
        input_numpy_size = get_numpy_size(input_tensor)
        input_bytes = input_numpy.tobytes()

        send_data = start + b'@#$%' + end + b'@#$%' + input_numpy_size + b'@#$%' + input_bytes
        # 发送长度
        send_data_len = str(len(send_data)).encode(encoding="UTF-8")
        # print("send_data长度：", len(send_data))
        self.datanode_socket.send(send_data_len)
        time.sleep(0.01)
        # 发送数据
        self.datanode_socket.sendall(send_data)
        # print ("datanode_socket 数据发送完成")

    def datanode_recv_data(self):
        # 先接受长度，再接收数据。
        data_total_len = b''
        while True:
            data = self.datanode_socket.recv(1024)
            if len(data) != 0:
                data_total_len = data
                break
        print ("DataNode recv data length: ", str(data_total_len, encoding='utf-8'))
        data_total_len = int(str(data_total_len, encoding="UTF-8"))
        print("DataNode recv data length: ", data_total_len)
        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.datanode_socket.recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp

        # print ("最终接受数据的长度：", len(recv_data), recv_data_len)
        split_list = recv_data.split(b'@#$%')
        start = int(str(split_list[0], encoding="UTF-8"))
        end = int(str(split_list[1], encoding="UTF-8"))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        # print ("recv_numpy_size: ", recv_numpy_size)
        recv_numpy = np.frombuffer(split_list[3], dtype = np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape = recv_numpy_size)
        recv_tensor = torch.from_numpy(recv_numpy)
        return start, end, recv_tensor

    def close(self):
        self.datanode_socket.close()

def get_recv_tensor_size(split_list_bytes):
    split_str = str(split_list_bytes, encoding="UTF-8").split("*")
    # print ("split_str:", split_str)
    recv_numpy_size = []
    for i ,value in enumerate(split_str):
        recv_numpy_size.append(int(value))
    # print ("recv_numpy_size:", type(tuple(recv_numpy_size)))
    return tuple(recv_numpy_size)

def get_numpy_size(input_tensor):

    size_list = list(input_tensor.size())
    input_numpy_size = ""
    length = len(size_list)
    for i, value in enumerate(size_list):
        if i == length - 1:
            input_numpy_size += str(value)
        else:
            input_numpy_size += str(value)
            input_numpy_size += "*"
    return input_numpy_size.encode(encoding="UTF-8")



# 之前的备份
# class Network_init_namenode():
#     def __init__(self, namenode_num = 1, datanode_num = 1):
#         super(Network_init_namenode, self).__init__()
#         print ("NameNode 开始初始化")
#         self.datanode_num = datanode_num
#         self.client_socket = []
#         if (datanode_num >= 1):
#             for hostname in range(datanode_num):
#                 tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 tcp_client_socket.connect((datanode_ip[hostname], datanode_port[hostname]))
#                 hello_world = "Hello DataNode "+ str(hostname) + ", I'm NameNode"
#                 tcp_client_socket.send(hello_world.encode())
#
#                 recv_data_test = tcp_client_socket.recv(1024)
#                 print (str(recv_data_test, encoding="UTF-8"))
#                 self.client_socket.append(tcp_client_socket)
#         print ("NameNode 初始化完成")
#
#     def namenode_send_data(self, i, input_tensor, start, end):
#         # 先发送数据长度，再发送数据
#         input_numpy = input_tensor.detach().numpy()
#         start = str(start).encode(encoding='utf-8')
#         end = str(end).encode(encoding='utf-8')
#         input_numpy_size = get_numpy_size(input_tensor)
#         input_bytes = input_numpy.tostring()
#         send_data = start + b':::' + end + b':::' + input_numpy_size + b':::' + input_bytes
#         # 发送长度
#         send_data_len = str(len(send_data)).encode(encoding='utf-8')
#         # print("send_data长度：", len(send_data))
#         self.client_socket[i].send(send_data_len)
#         # 发送数据
#         self.client_socket[i].sendall(send_data)
#         # print("namenode socket 数据发送完成")
#         # print("send_return_info: ", send_return_info)
#
#     def namenode_recv_data(self, i):
#         # 先接收数据长度，再接收数据
#         # 接收的数据：start 0，end 0， numpy_size, tensor
#         data_total_len = self.client_socket[i].recv(1024)
#         data_total_len = int(str(data_total_len, encoding='utf-8'))
#         recv_data_len = 0
#         recv_data = b''
#         while recv_data_len < data_total_len:
#             recv_data_temp = self.client_socket[i].recv(10240)
#             recv_data_len += len(recv_data_temp)
#             recv_data += recv_data_temp
#         # print("最终接受数据的长度：", len(recv_data), recv_data_len)
#
#         split_list = recv_data.split(b':::')
#         start = int(str(split_list[0], encoding="utf-8"))
#         end = int(str(split_list[1], encoding="utf-8"))
#
#         recv_numpy_size = get_recv_tensor_size(split_list[2])
#         # print("recv_numpy_size: ", recv_numpy_size)
#         recv_numpy = np.fromstring(split_list[3], dtype=np.float32)
#         recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
#         recv_tensor = torch.from_numpy(recv_numpy)
#         return start, end, recv_tensor
#
#     def close(self, i):
#         self.client_socket[i].close()
#     def close_all(self):
#         for i in range(self.datanode_num):
#             self.client_socket[i].close()
#
# class Network_init_datanode():
#     def __init__(self, namenode_num = 1, datanode_num = 3, hostname = 0):
#         super(Network_init_datanode, self).__init__()
#         print ("DataNode %d 开始初始化" % hostname )
#         # 创建服务器 socket
#         tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         tcp_server_socket.bind((datanode_ip[hostname], datanode_port[hostname]))
#         tcp_server_socket.listen(2)
#         # 得到client socket
#         self.datanode_socket, client_addr = tcp_server_socket.accept()
#         recv_data_test = self.datanode_socket.recv(1024)
#         # 简单的测试
#         print (str(recv_data_test, encoding="UTF-8"))
#         # 发送数据测试
#         hello_world = "Hello NameNode, I have received your hello world, I'm DateNode " + str(hostname)
#         self.datanode_socket.send(hello_world.encode())
#         print("DataNode %d 初始化完成" % hostname)
#
#     def datanode_send_data(self, input_tensor, start=0, end=0):
#         # 先发送数据长度，再发送数据
#         # print ("datanode_socket 数据发送开始")
#         input_numpy = input_tensor.detach().numpy()
#
#         start = str(start).encode(encoding='utf-8')
#         end = str(end).encode(encoding='utf-8')
#         input_numpy_size = get_numpy_size(input_tensor)
#         input_bytes = input_numpy.tostring()
#
#         send_data = start + b':::' + end + b':::' + input_numpy_size + b':::' + input_bytes
#         # 发送长度
#         send_data_len = str(len(send_data)).encode(encoding='utf-8')
#         # print("send_data长度：", len(send_data))
#         self.datanode_socket.send(send_data_len)
#         # 发送数据
#         self.datanode_socket.sendall(send_data)
#         # print ("datanode_socket 数据发送完成")
#
#     def datanode_recv_data(self):
#         # 先接受长度，再接收数据。
#         data_total_len = b''
#         while True:
#             data = self.datanode_socket.recv(1024)
#             if len(data) != 0:
#                 data_total_len = data
#                 break
#         # print ("DataNode recv data length: ", str(data_total_len, encoding='utf-8'))
#         data_total_len = int(str(data_total_len, encoding='utf-8'))
#         print("DataNode recv data length: ", data_total_len)
#         recv_data_len = 0
#         recv_data = b''
#         while recv_data_len < data_total_len:
#             recv_data_temp = self.datanode_socket.recv(10240)
#             recv_data_len += len(recv_data_temp)
#             recv_data += recv_data_temp
#
#         # print ("最终接受数据的长度：", len(recv_data), recv_data_len)
#         split_list = recv_data.split(b':::')
#         start = int(str(split_list[0], encoding="utf-8"))
#         end = int(str(split_list[1], encoding="utf-8"))
#
#         recv_numpy_size = get_recv_tensor_size(split_list[2])
#         # print ("recv_numpy_size: ", recv_numpy_size)
#         recv_numpy = np.fromstring(split_list[3], dtype = np.float32)
#         recv_numpy = np.reshape(recv_numpy, newshape = recv_numpy_size)
#         recv_tensor = torch.from_numpy(recv_numpy)
#         return start, end, recv_tensor
#
#     def close(self):
#         self.datanode_socket.close()
#
# def get_recv_tensor_size(split_list_bytes):
#     split_str = str(split_list_bytes, encoding='utf-8').split("*")
#     # print ("split_str:", split_str)
#     recv_numpy_size = []
#     for i ,value in enumerate(split_str):
#         recv_numpy_size.append(int(value))
#     # print ("recv_numpy_size:", type(tuple(recv_numpy_size)))
#     return tuple(recv_numpy_size)
#
# def get_numpy_size(input_tensor):
#
#     size_list = list(input_tensor.size())
#     input_numpy_size = ""
#     length = len(size_list)
#     for i, value in enumerate(size_list):
#         if i == length - 1:
#             input_numpy_size += str(value)
#         else:
#             input_numpy_size += str(value)
#             input_numpy_size += "*"
#     return input_numpy_size.encode(encoding="utf-8")
