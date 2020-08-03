import sys
sys.path.append("../..")
sys.path.append("..")

import torch, time
import torch.nn as nn
import numpy as np
from thop import profile
import matplotlib.pyplot as plt

def get_liner_model(X = [], Y = []):
    length = len(X)
    X_sum = 0.0
    Y_sum = 0.0
    X2sum = 0.0
    X_Y = 0.0
    for i in range(length):
        X_sum += X[i]
        Y_sum += Y[i]
        X_Y += X[i] * Y[i]
        X2sum += X[i]**2
    k = (X_sum * Y_sum / length -X_Y ) / (X_sum**2 / length - X2sum)
    b = (Y_sum - k * X_sum) / length
    print('the line is y = %f * x + %f' % (k, b))

# 测试包括Conv 、 pool 、 fully
def conv_test():

    # 二维数据记录FLOPS与其对应的运行时间
    # 测试 100 组数据, 取平均
    flops_record = []
    time_record = []
    chanel_list = [[3, 64], [64, 64], [64, 128], [128, 128], [128, 256], [256, 256], [256, 512], [512, 512]]
    # 调节模型的参数(in_chanel, out_chanel, width)，得到新的模型
    for chanel in chanel_list:
        in_chanel = chanel[0]
        out_chanel = chanel[1]
        for output_size in range(224, 0, -7):
            # 模型的输入
            input_size = output_size
            input = torch.randn(1, in_chanel, input_size, input_size)

            model = Conv(input_size = input_size, output_size = output_size, kernel_size = 3, in_chanel = in_chanel,
                         out_chanel = out_chanel, init_weights = True)
            counts = 1
            start_time = time.time()
            # 模型开始计算
            for i in range(counts):
                output = model(input)
            end_time = time.time()
            one_used_time = (end_time - start_time) / counts
            flops = model.get_flops()
            # 单位分别是：M， 和 ms（便于运算）
            flops_record.append(flops / (10 ** 6))
            time_record.append(one_used_time * 1000)
            # flops_record.append(flops)
            # time_record.append(one_used_time)
        print ("测试完成一组数据")
    # 自变量 x
    print( flops_record )
    # 因变量 y
    print( time_record )
    plt.plot(flops_record, time_record, 'o')
    plt.show()
    # 求回归方程
    get_liner_model(flops_record, time_record)

def fully_test():
    # 二维数据记录FLOPS与其对应的运行时间
    # 测试 100 组数据, 取平均
    it = 15
    jt = 15
    flops_and_time_record = np.zeros(( (it+1) * (jt+1) , 2), dtype=np.float)
    flops_record = []
    time_record = []
    for i in range(it+1):
        input_size = int(100 + (4096-100)/it * i)
        for j in range(jt+1):
            output_size = int(100 + (4096-100)/jt * j)
            # print ("input_size: %d, output_size: %d " % (input_size, output_size))
            # 模型的输入
            input = torch.randn(input_size)
            model = Fully_layer(input_size = input_size, output_size = output_size, init_weights=True )
            # 测试一百组取平均值
            counts = 1
            start_time = time.time()
            for m in range(counts):
                output = model(input)

            end_time = time.time()
            one_used_time = (end_time - start_time) / counts
            flops = model.get_flops()
            # 单位分别是：M， 和 ms（便于运算）
            # flops_and_time_record[i * (jt + 1) + j][0] = flops / (10 ** 6)
            # flops_and_time_record[i * (jt + 1) + j][1] = one_used_time * (10**3)
            flops_record.append(flops / (10 ** 6))
            time_record.append(one_used_time * 1000)

    print(flops_record)
    print(time_record)
    get_liner_model(flops_record, time_record)
    plt.plot(flops_record, time_record, 'o')
    plt.show()





class Fully_layer(nn.Module):
    def __init__(self, input_size = 4096, output_size = 4096, init_weights = True ):
        super(Fully_layer, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(True),
        )
        self.flops = (2 * input_size - 1) * output_size
        if init_weights:
            self._initialize_weights()
    def get_flops(self):
        return self.flops

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class AvgPool2d(nn.Module):
    def __init__(self, input_size = 224, output_size = 224, kernel_size = 2, in_chanel = 64, out_chanel=64, init_weights=True ):
        super(AvgPool2d, self).__init__()
        self.features_1_3 = nn.Sequential(
            # nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
        )
        # 2 * 2 de 平均池化层可以理解为2*2的
        self.flops = 2 * output_size * output_size * (in_chanel * kernel_size * kernel_size + 1) * out_chanel
        if init_weights:
            self._initialize_weights()
    def get_flops(self):
        return self.flops

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Conv(nn.Module):
    def __init__(self, input_size = 224, output_size = 224, kernel_size = 3, in_chanel = 3, out_chanel = 64, init_weights=True ):
        super(Conv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chanel, out_chanel, kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # self.flops = ( (kernel_size * kernel_size * in_chanel) * out_chanel + out_chanel ) * (output_size * output_size)
        self.flops = 2 * output_size * output_size * (in_chanel * kernel_size * kernel_size + 1) * out_chanel
        if init_weights:
            self._initialize_weights()
    def get_flops(self):
        return self.flops

    def forward(self, x):
        x = self.features(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



if __name__ == "__main__":

    # Conv测试
    conv_test()
    # Fully测试
    # fully_test()



    # conv_model = Conv(input_size = 224, output_size = 224, kernel_size = 3, in_chanel = 3, out_chanel = 64,)
    # print ("Conv Flops: %.2fM" % (conv_model.get_flops()/(10**6)))


    # fully_model = Fully_layer()
    # print ("Fully FLOPS: %d" % (fully_model.get_flops()/(10**6)))

    # 测试线性函数
    # x = [1, 2, 4, 5, 6, 7, 3 ,8]
    # y = [5, 7, 11, 13, 15, 17, 9, 18]
    # get_liner_model(x, y)
