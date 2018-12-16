import os
from power import power_m
import numpy as np
import re

def split_stats(path: str,file:str):
    """
    将文件切割为单个stats.txt

    :param path: 路径
    :param file: 文件名
    :return:
    """
    with open(path +'/'+file, "r", encoding='utf-8') as f:
        count = 0
        data_path = path + '/stats'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        line = f.readline()
        while line:
            while line and 'Begin' not in line:
                line = f.readline()
            if 'Begin' in line:
                file_name=data_path+'/stats_'+str(count)+'.txt'
                with open(file_name, 'w', encoding='utf-8') as ff:
                    while 'End' not in line:
                        ff.write(line)
                        line = f.readline()
                    ff.write(line)
                    line = f.readline()
                print('Generating stats_'+str(count)+'.txt')
                count += 1

def init(data_path,program):
    """
    读取x86数据，存在data_path下

    :param data_path: 存储路径
    :param program: 程序名
    :return:
    """
    path="/home/lianghong/data/"+program+"/data"
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

    mask=[1,1,1,0,1,1,1,0]
    masked_event=[]
    for i in range(len(mask)):
        if mask[i]:
            masked_event.append(avg_event[i])
    masked_event=np.array(masked_event).T
    power=power.T
    file=data_path+"/"+program+"/data/events"
    with open(file,"w",encoding="utf-8") as f:
        length=len(power)
        for i in range(length):
            f.write("%10.2d %10.2d %10.2d %10.2d %10.2d %10.2d %10.2f\n" %(masked_event[i][0],masked_event[i][1],masked_event[i][2],masked_event[i][3],masked_event[i][4],masked_event[i][5],power[i]))

def write_data_file(stats,config,template,save):
    """
    从stats文件读取性能事件，计算功耗并存储到指定文件

    :param stats: stats目录
    :param config: config.json文件
    :param template: mcpat模板文件
    :param save: 数据存储路径
    :return:
    """
    file_list = os.listdir(stats)
    result=[]
    for file in file_list:
        r = [0 for i in range(9)]
        with open(stats + "/" + file, "r", encoding="utf-8") as f:
            line = f.readline()
            while line != "\n":
                t = line.split()
                if "sim_insts" in line:
                    r[0] = int(t[1])
                elif "system.cpu" in line:
                    for c in range(4):
                        prefix = "system.cpu" + str(c) + "."
                        if prefix + "Branches" in line:
                            r[1] += int(t[1])
                        elif prefix + "icache.overall_accesses::total" in line:
                            r[3] += int(t[1])
                        elif prefix + "dcache.overall_accesses::total" in line:
                            r[4] += int(t[1])
                        elif prefix + "num_mem_refs" in line:
                            r[5] += int(t[1])
                        elif prefix + "numCycles" in line:
                            r[7] += int(t[1])
                elif "system.l2.overall_accesses::total" in line:
                    r[2] = int(t[1])
                elif "system.l2.writebacks::total" in line:
                    r[6] = int(t[1])
                line = f.readline()

        home = "/home/lianghong/Desktop/GraduateData/research1/run"
        order = re.findall("\\d+", file)[0]
        # print(order)
        os.system(
            home + "/GEM5ToMcPAT.py" + " " + stats + "/" + file + " " + config + " " + template + " -o " + home + "/power_test/mcpat_input/mcpat_" + order + ".xml")
        mcpat_command = "/home/lianghong/Downloads/mcpat/mcpat"
        power = os.popen(mcpat_command + " -infile " + home + "/power_test/mcpat_input/mcpat_" + order + ".xml -print_level 1")
        lines = power.readlines()
        for line in lines:
            if "Peak Power" in line:
                t = line.split()
                r[8] = float(t[3])
                break
        power.close()
        result.append(r)

    with open(save,"w",encoding="utf-8") as ff:
        size=len(result)
        for i in range(size):
            ff.write("%10d %10d %10d %10d %10d %10d %10d %10d %10.2f\n" % (result[i][0],result[i][1],result[i][2],result[i][3],result[i][4],result[i][5],result[i][6],result[i][7],result[i][8]))





if __name__=='__main__':
    # split_stats("/home/lianghong/Desktop/GraduateData/research1/run/power_test","stats.txt")
    prefix="/home/lianghong/Desktop/GraduateData/research1/run"
    write_data_file(prefix+"/power_test/stats",prefix+"/power_test/config.json",prefix+"/power_test/template/arm.xml",prefix+"/power_test/cov")