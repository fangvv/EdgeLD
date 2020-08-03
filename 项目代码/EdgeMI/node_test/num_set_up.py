
datanode_num_temp = 1
class Num_set_up(object):
    def __int__(self ):
        self.namenode_num = 1
        self.datanode_num = datanode_num_temp
    def set_namenode_num(self, num):
        self.namenode_num = num
    def get_namenode_num(self):
        self.set_namenode_num(1)
        return self.namenode_num
    def set_datanode_num(self, num):
        self.datanode_num = num
    def get_datanode_num(self):
        self.set_datanode_num(datanode_num_temp)
        return self.datanode_num
