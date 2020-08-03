import socket, time, json, torch, six
import torch.nn as nn
import numpy as np


def main():
    # 1. 创建tcp的套接字
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 2. 链接服务器
    tcp_socket.connect(("127.0.0.1", 10000))
    # 3. 发送数据/接收数据

    input_tensor = torch.rand(1, 2, 3, 4)
    print (input_tensor)
    input_numpy = input_tensor.numpy()
    input_numpy_size = np.asarray(input_numpy.shape, dtype=np.int8)

    start = six.int2byte(1)
    end = six.int2byte(13)
    input_numpy_size = input_numpy_size.tostring()
    input_bytes = input_numpy.tostring()
    send_data = start + b'::' + end + b'::' + input_numpy_size+ b'::' + input_bytes
    tcp_socket.send(send_data)
    time.sleep(1)

    # 4. 关闭套接字
    tcp_socket.close()


if __name__ == "__main__":
    main()