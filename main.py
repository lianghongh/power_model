from nn import Network
import numpy as np
import tensorflow as tf
from power import mcpat
import os


def read_data(path,program):
    event,power=[],[]
    with open(path+"/"+program+"/data/events","r",encoding="utf-8") as f:
        line=f.readline()
        while line:
            d=line.split()
            tmp=[]
            tmp.append(float(d[0]))
            tmp.append(float(d[1]))
            tmp.append(float(d[2]))
            tmp.append(float(d[3]))
            tmp.append(float(d[4]))
            tmp.append(float(d[5]))
            event.append(tmp)
            power.append(float(d[6]))
            line=f.readline()
    return np.array(event,dtype=np.float32),np.array([power],dtype=np.float32).T


def benchmark(data_path, act_func, save_path):
    if not os.path.exists(data_path+"/"+program+"/data"):
        os.makedirs(data_path+"/"+program+"/data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mcpat.init(data_path,program)
    event,power=read_data(data_path,program)

    # 测试线性模型
    file_name= save_path + "/linear/prediction_acc"
    if os.path.exists(file_name):
        os.remove(file_name)

    model = Network.linear_model_train(event, power, path=save_path, program_name=program)
    Network.graph(model(event), power, program_name=program, save_path=save_path + "/linear/" + program)

    # 测试单隐含层模型
    file_name= save_path + "/onelayer/prediction_acc"
    if os.path.exists(file_name):
        os.remove(file_name)
    for node in range(4,14):
        model=Network.onelayer_model(event, power, layer_node=node, act_func=act_func, path=save_path, program_name=program)
        Network.graph(model(event), power, program_name=program, save_path=save_path + "/onelayer/" + program + "_" + str(node))

    # 测试双隐含层模型
    file_name= save_path + "/twolayer/prediction_acc"
    if os.path.exists(file_name):
        os.remove(file_name)
    for node1 in range(4,14):
        for node2 in range(4,14):
            model=Network.twolayer_model(event, power, layer1_node=node1, layer2_node=node2, act_func=act_func, path=save_path, program_name=program)
            Network.graph(model(event), power, program_name=program, save_path=save_path + "/twolayer/" + program + "_" + str(node1) + "_" + str(node2))


if __name__=='__main__':
    program_list=["blackscholes","canneal","cholesky","facesim","fluidanimate","freqmine","x264","fft"]

    for program in program_list:
        path = "/home/lianghong/Desktop/GraduateData/research1/data"
        figure="/home/lianghong/Desktop/GraduateData/research1/data/" + program + "/figure"
        benchmark(path,act_func=tf.nn.relu,save_path=figure)


    # l=cov.get_spearman_list("/home/lianghong/data/events_data","/home/lianghong/data/power_data",0.1,300)
    # for i in l:
    #     print(i)

