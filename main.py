from power import power_m
from nn import Network
import numpy as np


if __name__=='__main__':
    prefix='/home/lianghong/data/result/blackscholes/data'
    power=np.array([power_m.get_power_list(prefix+'/power',0.1)[0]],dtype=np.float32)
    events=[]
    for i in range(8):
        events.append(power_m.get_event_list(prefix+'/cpu_'+str(i),0.1))
    length=len(events[0][0])
    avg_event=np.array([[0 for i in range(length)] for j in range(11)],dtype=np.float32)

    for i in range(8):
        for j in range(11):
            avg_event[j]=avg_event[j]+np.array(events[i][j])

    for i in range(11):
        avg_event[i]=avg_event[i]/8

    model=Network.linear_model_train(avg_event[:8].T,power.T)
    Network.graph(model(avg_event[:8].T),power.T)

    # l=cov.get_spearman_list("/home/lianghong/data/events_data","/home/lianghong/data/power_data",0.1,300)
    # for i in l:
    #     print(i)

