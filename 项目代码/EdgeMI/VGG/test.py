import sys
sys.path.append("../..")
sys.path.append("..")

def con(x):
    a = 6.24e-5 * x + 1.97e-2
    return a
def full(x):
    a = 5.12e-4 * x + 8.28e-5
    return a
FLOPs_list_conv = [179.8, 3705.8, 1852.9, 3702.6, 1851.3, 3701, 3701, 1850.5, 3700.2, 3700.2, 925, 925, 925]
print (sum(FLOPs_list_conv))
FLOPs_list_full = [205, 33.5, 8.2]
for i in  FLOPs_list_conv:
    print (con(i))

for i in FLOPs_list_full:
    print(full(i))







import torch, time, socket, json, six
import torch.nn as nn
from threading import Thread
import numpy as np

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix
# X = sparse_random(4096, 4096, density = 0.01, format = 'csr', random_state = 42)
# svd = TruncatedSVD(n_components = 5, n_iter = 7, random_state = 42)
# # svd.fit(X)
# svd.fit_transform(X)
# print(svd.explained_variance_ratio_)
# print(svd.explained_variance_ratio_.sum())
# print(svd.singular_values_)
# X_new = svd.transform(X)

# from sklearn.datasets import load_iris
# iris = load_iris()
# iris_data = iris.data
#
# svd = TruncatedSVD(2)
# iris_transformed = svd.fit_transform(iris_data)
# print ( iris_data[:5] )





# input = torch.rand(1, 3, 64, 64)
# size_list = list(input.size())
# input_numpy_size = ""
# len = len(size_list)
# for i, value in enumerate(size_list):
#     if i == len-1:
#         input_numpy_size += str(value)
#     else:
#         input_numpy_size += str(value)
#         input_numpy_size += "*"
# print (input_numpy_size)


# def recv_inference_data(recv_data):
#     '''
#     :param recv_data: socket接收到的数据
#     :return: start, end , recv_tensor
#     '''
#     split_list = recv_data.split(b'::')
#     start = six.byte2int(split_list[0])
#     end = six.byte2int(split_list[1])
#     recv_numpy_size = tuple(np.fromstring(split_list[2], dtype=np.int8))
#     recv_numpy = np.fromstring(split_list[3], dtype=np.float32)
#     recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
#     recv_tensor = torch.from_numpy(recv_numpy)
#
#     return start, end , recv_tensor
#
#
#     # 发送
#     socket.send(send_data)
#     # 判断是否发送成功(暂时不考虑)
#
# def send_inference_data(socket, input_tensor, start, end):
#     '''
#     socket, 服务器连接
#     input_tensor, start_layer, end_layer
#     :return: part inference result
#     假设datanode有DNN模型
#     '''
#     input_numpy = input_tensor.numpy()
#
#     start = six.int2byte(start)
#     end = six.int2byte(end)
#     input_numpy_size = (np.asarray(input_numpy.shape, dtype=np.int8)).tostring()
#     input_bytes = input_numpy.tostring()
#     send_data = start + b'::' + end + b'::' + input_numpy_size + b'::' + input_bytes
#     # 发送
#     socket.send(send_data)
#     # 判断是否发送成功(暂时不考虑)
#
# def ndarray2str(array):
#     # Convert the numpy array to string
#     return array.tostring()
#
# def str2ndarray(str):
#     # array = np.fromstring(str.data, dtype=np.float32)
#     return np.fromstring(str.data, dtype=np.float32)
#
#
# if __name__ == '__main__':
#
#     VGG16_model = mydefine_VGG16.model()
#     start = six.int2byte(2)
#     end = six.int2byte(3)
#     input = torch.rand(4)
#     input_numpy = input.numpy()
#     input_tostring = input_numpy.tostring()
#     # 传输的字节合并
#     input_bytes = start + b':::' + end + b':::' + input_tostring
#     print (start)
#     print (end)
#     print (input_tostring)
#     print (input_bytes)
#     split_list = input_bytes.split(b':::')
#     temp = np.fromstring(input_numpy.tostring(), dtype=np.float32)
#
#
#     print(temp == input_numpy)
#     print(type(input_tostring))
#     print(torch.equal(input, torch.from_numpy(temp)))
#
#
#     data_dict = {}
#     # data_dict['input_tensor'] = input_string
#     data_dict['input_tensor'] = input_tostring
#     data_dict['start'] = 1
#     data_dict['end'] = 2
#
#     send_data = json.dumps(data_dict)
#     # 往回转化
#     get_data = json.loads(send_data)
#     get_tensor_string = get_data['input_tensor']
#     # 该转换出现问题
#     get_numpy = np.array(get_tensor_string, dtype=np.float32)
#
#     if get_numpy == input_numpy:
#         print("True")
#     else:
#         print("False")
#
#     new_tensor = torch.from_numpy(get_numpy)
#     print (new_tensor)
#
#
#
#
#
#
#
#
#     count = 100
    # start_time = time.time()
    # for i in range(count):
        # print (i)
        # start_time = time.time()
        # output = VGG16_model(input, 1, 13)
        # output = VGG16_model(input, 1, 6)
        # output = VGG16_model(output, 7, 13)

        # output = VGG16_model(input, 1, 4)
        # output = VGG16_model(output, 5, 8)
        # output = VGG16_model(output, 9, 13)
        # end_time = time.time()
        # print(output.size())
        # print("Used time: ", end_time - start_time)
    # end_time = time.time()
    # print("One inference used time: %.2fs" %((end_time - start_time)/count) )







# model = vgg.vgg16(pretrained=True)
# print (model)
# print (model.state_dict())
# for name in model.state_dict():
#    print(name)

# params = list(model.named_parameters()) # get the index by debuging
# print(params[3][0]) # name
# print(params[2][1].data) # data

# for name in model.state_dict():
#    print(name)
#    a = model.state_dict()[name]
#    print(a.size())

# temp = torch.rand(128, 224, 224, 3)
# a, b, c, d = temp.size()
# print (a, b, c, d)
# # print(temp_1.size())
# temp_2 = temp[:, :, 0:112, :]
# print (temp_2.size())
# temp_3 = temp[:, :, 112:224, :]
# print (temp_3.size())
# x = torch.cat((temp_2, temp_3), 2)
# print (torch.equal(temp, x))
# class model(nn.Module):
#     def __init__(self, num_classes=10, init_weights=True):
#         super(model, self).__init__()
#         self.features_1 = nn.Sequential(
#             nn.Conv2d(3, 32, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.features_2 = nn.Sequential(
#             nn.Conv2d(32, 32, 3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features_1(x)
#         x = self.features_2(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

# input = torch.rand(10, 3, 224, 224)
# a, b, c, d = input.size()
# layer_num = 3
#
# input_0 = input[:, :, :, 0:int(d/2+layer_num)]
# input_1 = input[:, :, :, int(d/2-layer_num):int(d)]
#
# model_temp = model()
# output = model_temp(input)
# output_0 = model_temp(input_0)
# output_1 = model_temp(input_1)

# print(output_0.size())
# print(output_1.size())
# print(output[1, 0, 0, 111])
# print(output_0[1, 0, 0, 111])
# print(output_1[1, 0, 0, 1])

# x = torch.cat((output_0[:, :, :, 0:int(d/2)], output_1[:, :, :, int(layer_num):int(d/2+layer_num)]), 3)
# print (x.size())
# print (torch.equal(output, x))

# 测时间
# count = 10
# start_time = time.time()
# for i in range(count):
#     output = model_temp(input)
# end_time = time.time()
# print ("Used time:", end_time-start_time)
#
# start_time = time.time()
# for i in range(count):
#     output_0 = model_temp(input_0)
#     output_1 = model_temp(input_1)
#     # x = torch.cat((output_0[:, :, :, 0:112], output_1[:, :, :, 3:115]), 3)
# end_time = time.time()
# print ("Used time:", end_time-start_time)


