from power import power_m
from nn import Network
import numpy as np
import tensorflow as tf
import os

def benchmark(program_path,act_func,path):
    if not os.path.exists(path):
        os.mkdir(path)

    power = np.array([power_m.get_power_list(program_path + '/power', 0.1)[0]], dtype=np.float32) / 10 * 2
    events = []
    for i in range(8):
        events.append(power_m.get_event_list(program_path + '/cpu_' + str(i), 0.1))
    length = len(events[0][0])
    avg_event = np.array([[0 for i in range(length)] for j in range(11)], dtype=np.float32)

    for i in range(8):
        for j in range(11):
            avg_event[j] = avg_event[j] + np.array(events[i][j])

    for i in range(11):
        avg_event[i] = avg_event[i] / 8

    # 测试线性模型
    model = Network.linear_model_train(avg_event[:8].T, power.T)
    Network.graph(model(avg_event[:8].T), power.T, program_name=program, save_path=path+"/linear")

    # 测试单隐含层模型
    file_name=path+"/onelayer/prediction_acc"
    if os.path.exists(file_name):
        os.remove(file_name)
    for node in range(4,14):
        model=Network.onelayer_model(avg_event[:8].T,power.T,layer_node=node,act_func=act_func,path=path,program_name=program)
        Network.graph(model(avg_event[:8].T), power.T, program_name=program, save_path=path+"/onelayer/"+program+"_"+str(node))

    # 测试双隐含层模型
    file_name=path+"/twolayer/prediction_acc"
    if os.path.exists(file_name):
        os.remove(file_name)
    for node1 in range(4,14):
        for node2 in range(4,14):
            model=Network.twolayer_model(avg_event[:8].T,power.T,layer1_node=node1,layer2_node=node2,act_func=act_func,path=path,program_name=program)
            Network.graph(model(avg_event[:8].T), power.T, program_name=program, save_path=path+"/twolayer/"+program+"_"+str(node1)+"_"+str(node2))

if __name__=='__main__':
    program="blackscholes"
    path = "/home/lianghong/data/result/" + program + "/data"
    benchmark(path,act_func=tf.nn.relu,path="/home/lianghong/pic")

    # l=cov.get_spearman_list("/home/lianghong/data/events_data","/home/lianghong/data/power_data",0.1,300)
    # for i in l:
    #     print(i)

