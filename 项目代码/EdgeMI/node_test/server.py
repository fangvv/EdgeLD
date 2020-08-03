import socket, json, torch, six
import torch.nn as nn
import numpy as np

def main():
    # 1. 买个手机(创建套接字 socket)
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 2. 插入手机卡(绑定本地信息 bind)
    tcp_server_socket.bind(("127.0.0.1", 10000))

    # 3. 将手机设置为正常的 响铃模式(让默认的套接字由主动变为被动 listen)
    tcp_server_socket.listen(128)

    print("-----1----")
    # 4. 等待别人的电话到来(等待客户端的链接 accept)
    new_client_socket, client_addr = tcp_server_socket.accept()
    print("-----2----")

    print(client_addr)
    # 接收客户端发送过来的请求
    recv_data = new_client_socket.recv(1024)
    split_list = recv_data.split(b'::')
    start = six.byte2int(split_list[0])
    end = six.byte2int(split_list[1])
    recv_numpy_size = tuple(np.fromstring(split_list[2], dtype=np.int8))
    recv_numpy = np.fromstring(split_list[3], dtype=np.float32)
    recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
    print (recv_numpy)
    recv_tensor = torch.from_numpy(recv_numpy)
    print (recv_tensor)

    # 回送一部分数据给客户端
    # new_client_socket.send("TCP Comm shutdown !!!!!!!!!!!!!!".encode("utf-8"))

    # 关闭套接字
    new_client_socket.close()
    tcp_server_socket.close()



if __name__ == "__main__":
    main()