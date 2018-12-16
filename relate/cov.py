import numpy as np
from power import power_m
from sklearn.preprocessing import MinMaxScaler


def pearson(v1, v2):
    n = len(v1)
    sum1 = sum(v1)
    sum2 = sum(v2)
    sum_pow1 = sum(i ** 2 for i in v1)
    sum_pow2 = sum(i ** 2 for i in v2)
    p_sum = np.dot(v1, v2)
    num = p_sum - (sum1 * sum2 / n)
    d = np.sqrt((sum_pow1 - sum1 ** 2 / n) * (sum_pow2 - sum2 ** 2 / n))
    if d == 0:
        return 0
    return num / d

def spearman(v1, v2):
    n = len(v1)
    sv1 = list(zip(v1, [i for i in range(1, n + 1)]))
    sv2 = list(zip(v2, [i for i in range(1, n + 1)]))
    ssv1 = sorted(sv1, key=lambda x: x[0], reverse=True)
    ssv2 = sorted(sv2, key=lambda x: x[0], reverse=True)
    d1 = np.array([j + 1 for i in range(n) for j in range(n) if sv1[i][1] == ssv1[j][1]])
    d2 = np.array([j + 1 for i in range(n) for j in range(n) if sv2[i][1] == ssv2[j][1]])
    return 1 - 6 * np.sum((d1 - d2) ** 2) / (n ** 3 - n)

def process(app_list,save_path):
    for app in app_list:
        path = "/home/lianghong/data/" + app + "/data"
        power = np.array([power_m.get_power_list(path + '/power', 0.1)[0]], dtype=np.float32) / 10 * 2
        events = []
        for i in range(8):
            events.append(power_m.get_event_list(path + '/cpu_' + str(i), 0.1))
        length = len(events[0][0])
        avg_event = np.array([[0 for i in range(length)] for j in range(11)], dtype=np.float32)

        for i in range(8):
            for j in range(11):
                avg_event[j] = avg_event[j] + np.array(events[i][j])

        for i in range(11):
            avg_event[i] = avg_event[i] / 8

        avg_event=avg_event.T
        power=power.T
        file=save_path+"/"+app+"/data/cov"
        with open(file,"w",encoding="utf-8") as f:
            length=len(power)
            for i in range(length):
                f.write("%10.2d %10.2d %10.2d %10.2d %10.2d %10.2d %10.2d %10.2d %10.2f\n" %(avg_event[i][0],avg_event[i][1],avg_event[i][2],avg_event[i][3],avg_event[i][4],avg_event[i][5],avg_event[i][6],avg_event[i][7],power[i]))
        print("%s cov completed!" %(app))


def cal_correlation(data_path):
    scalar=MinMaxScaler()
    event,power=[],[]
    with open(data_path,"r",encoding="utf-8") as f:
        line=f.readline()
        while line:
            d=line.split()
            t=[]
            t.append(float(d[0]))
            t.append(float(d[1]))
            t.append(float(d[2]))
            t.append(float(d[3]))
            t.append(float(d[4]))
            t.append(float(d[5]))
            t.append(float(d[6]))
            t.append(float(d[7]))
            event.append(t)
            power.append(float(d[8]))
            line=f.readline()
    event=np.array(event,dtype=np.float32).T
    power=np.array(power,dtype=np.float32).reshape(-1,1)
    r = []
    length = len(event)
    for i in range(length):
        r.append([i, spearman(scalar.fit_transform(event[i].reshape(-1,1)), scalar.fit_transform(power))])
    r.sort(key=lambda x:abs(x[1]),reverse=True)
    return r

def cov_analysis(app_list,event_length,k,save_path):
    """
    相关性分析

    :param app_list: app列表
    :param event_length: 事件总数
    :param k: 选k个事件
    :param save_path: 存储路径
    :return:
    """
    if k>event_length:
        print("k is too large!")
        return []
    if k==event_length:
        return [i for i in range(event_length)]
    counter=[[i,0] for i in range(event_length)]
    for app in app_list:
        path=save_path+"/"+app+"/data/cov"
        data=cal_correlation(path)
        for j in range(k):
            counter[data[j][0]][1]+=1
        print("%s cov analysis completed!" %(app))
    counter.sort(key=lambda x:x[1],reverse=True)
    with open(save_path+"/cov_analysis","w",encoding="utf-8") as f:
        for i in counter:
            f.write(str(i)+"\n")
    return counter