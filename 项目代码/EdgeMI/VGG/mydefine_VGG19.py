import sys
sys.path.append("../..")
sys.path.append("..")


import torch
import torch.nn as nn
import time

class VGG_model(nn.Module):
    def __init__(self, num_classes = 100, init_weights = True):
        super(VGG_model, self).__init__()
        self.module_list = []
        self.maxpool_layer = [3, 6, 10, 14, 21]

        # 定义 forward 中用到的 中间结果 和 最终结果
        self.middle_result = torch.rand(1, 1, 1, 1)
        self.final_result = torch.rand(1, 100)
        ##################################################
        self.features_1_1 = nn.Sequential(
            # 0 - 1
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features_1_2 = nn.Sequential(
            # 2 - 3
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )# 1*64*224*224
        self.features_1_3 = nn.Sequential(
            # 4 - 4
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # 1*64*112*112

        ##################################################
        self.features_2_1 = nn.Sequential(
            # 5 - 6
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features_2_2 = nn.Sequential(
            # 7 - 8
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )# 1*128*56*56
        self.features_2_3 = nn.Sequential(
            # 9 - 9
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # 1*128*56*56


        ##################################################
        self.features_3_1 = nn.Sequential(
            # 10 -11
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features_3_2 = nn.Sequential(
            # 12 -13
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features_3_3 = nn.Sequential(
            # 14 -15
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )# 1*256*28*28
        self.features_3_4 = nn.Sequential(
            # 14 -15
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )  # 1*256*28*28
        self.features_3_5 = nn.Sequential(
            # 16 -16
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # 1*256*28*28

        ##################################################
        self.features_4_1 = nn.Sequential(
            # 17 - 18
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features_4_2 = nn.Sequential(
            # 19 - 20
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features_4_3 = nn.Sequential(
            # 21 - 22
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )# 1*512*14*14
        self.features_4_4 = nn.Sequential(
            # 21 - 22
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )  # 1*512*14*14
        self.features_4_5 = nn.Sequential(
            # 23 - 23
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # 1*512*14*14

        ##################################################
        self.features_5_1 = nn.Sequential(
            # 24 - 25
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features_5_2 = nn.Sequential(
            # 26 - 27
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features_5_3 = nn.Sequential(
            # 28 - 29
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )# 1*512*7*7
        self.features_5_4 = nn.Sequential(
            # 28 - 29
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )  # 1*512*7*7
        self.features_5_5 = nn.Sequential(
            # 30 - 30
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # 1*512*7*7

        ##################################################
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier_1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        # module_list_list不是使用0下标，正式下标对应着层数
        self.module_list.append(self.features_1_1) # 填充，占位置

        # 以下为 “卷积层”
        self.module_list.append(self.features_1_1) #1
        self.module_list.append(self.features_1_2) #2
        self.module_list.append(self.features_1_3) #3
        self.module_list.append(self.features_2_1) #4
        self.module_list.append(self.features_2_2) #5
        self.module_list.append(self.features_2_3) #6
        self.module_list.append(self.features_3_1) #7
        self.module_list.append(self.features_3_2) #8
        self.module_list.append(self.features_3_3) #9
        self.module_list.append(self.features_3_4) #10
        self.module_list.append(self.features_3_5) #11
        self.module_list.append(self.features_4_1) #12
        self.module_list.append(self.features_4_2) #13
        self.module_list.append(self.features_4_3) #14
        self.module_list.append(self.features_4_4) #15
        self.module_list.append(self.features_4_5) #16
        self.module_list.append(self.features_5_1) #17
        self.module_list.append(self.features_5_2) #18
        self.module_list.append(self.features_5_3) #19
        self.module_list.append(self.features_5_4) #20
        self.module_list.append(self.features_5_5) #21
        self.module_conv_length = len(self.module_list) - 1
        # 以下为 “全连接层”
        self.module_list.append(self.classifier_1)  # 22
        self.module_list.append(self.classifier_2)  # 23
        self.module_list.append(self.classifier_3)  # 24
        # 统计 “卷积层” 和 “全连接层” 的长度
        self.module_total_length = len(self.module_list) - 1
        # 所有网络层参数初始化，包括 “卷积层”和“全连接层”
        if init_weights:
            self._initialize_weights()

    # input 输入， start开始层， end结束层（包括）。"全连接层" 暂时不写创新的地方
    def forward(self, x, start = 0, end = 0):
        if (start > end or start <= 0 or end > self.module_total_length):
            print ("输入参数有误，检查后重新输入")
            return x
        # 全部计算在“卷积层”
        elif start < self.module_conv_length and end <= self.module_conv_length:
            for i in range(start, end + 1, 1):
                x = self.module_list[i](x)
            return x
        # 全部计算在“全连接层”
        elif start > self.module_conv_length:
            for i in range(start, end + 1, 1):
                x = self.module_list[i](x)
            return x
        # 计算在“卷积层”和“全连接层”
        else:
            for i in range(start, end + 1, 1):
                x = self.module_list[i](x)
                if i == self.module_conv_length:
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                # print (x.size())
            return x


    # 获取“卷积层”长度
    def get_conv_length(self):
        return self.module_conv_length

    # 获取“卷积层”长度
    def get_total_length(self):
        return self.module_total_length
    # 获取池化层记录
    def get_maxpool_layer(self):
        return self.maxpool_layer

    # 参数初始化方法
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


if __name__ == '__main__':

    VGG19_model = VGG_model()
    input = torch.rand(1, 3, 224, 224)

    total_length = VGG19_model.get_total_length()
    count = 5
    start_time = time.time()
    for i in range(count):
        output = VGG19_model(input, 1, total_length)
    end_time = time.time()
    print("One inference used time: %.4fs" %((end_time - start_time)/count) )
