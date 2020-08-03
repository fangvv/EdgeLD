import sys
sys.path.append("../..")
sys.path.append("..")

# 记录 网络状态 和 计算能力,人工提前设定常量
class Network_And_Computing():
    def __init__(self):
        # 算力单位描述为 s/FLOPs
        self.computing_power = []
        self.computing_a = [6.24e-11, 6.24e-11, 6.24e-11, 6.24e-11, 3.24e-11, 6.24e-11, 6.24e-11]
        self.computing_b = [1.97e-2, 1.97e-2, 1.97e-2, 1.97e-2, 1.97e-2, 1.97e-2, 1.97e-2]
        for i in self.computing_a:
            # 倒数
            self.computing_power.append(1.0/i)
        # 网络单位为 bps
        self.network_state = [100e6, 10e6, 300e6, 1000e6, 50e6, 500e6, 100e6]
    def get_computing_a(self, datanode_num = 1):
        return self.computing_a[0: datanode_num]
    def get_computing_b(self, datanode_num = 1):
        return self.computing_b[0: datanode_num]

    def get_c(self):
        print (self.computing_power)

    def get_computing_power(self, datanode_num = 1):
        return self.computing_power[0 : datanode_num]
    def get_computing_power_normalization(self, datanode_num = 1):
        max_value = max(self.computing_power[0 : datanode_num])
        for i in range(datanode_num):
            self.computing_power[i] = self.computing_power[i]/max_value
        return self.computing_power[0 : datanode_num]

    def get_network_state(self, datanode_num = 1):
        return self.network_state[0 : datanode_num]
    def get_network_state_normalization(self, datanode_num=1):
        max_value = max(self.network_state[0: datanode_num])
        for i in range(datanode_num):
            self.network_state[i] = self.network_state[i] / max_value
        return self.network_state[0: datanode_num]


# if __name__ == "__main__":
#     temp = Network_And_Computing()
#     temp.get_c()



