import os
from power import power_m
import numpy as np


def split_stats(path: str):
    with open(path + '/stats.txt', "r", encoding='utf-8') as f:
        count = 0
        data_path = path + '/mcpat_input'
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
                print('Generating mcpat_'+str(count)+'.xml')
                count += 1

def init(data_path,program):
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

    mask=[0,1,1,1,1,1,1,0]
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


def fill_template(path,config_path,template_path):
    file_list=os.listdir(path)
    for file in file_list:
        os.system(path+"GEM5ToMcPAT.py "+path+"/"+file+" "+config_path+" "+template_path+" -o"+path+"_"+file+"_"+"mcpat.xml")



def compute_power(path,output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    file_list=os.listdir(path)
    with open(output_path+'/power.txt','w',encoding='utf-8') as f:
        for i in file_list:
            content=os.popen('/home/lianghong/Desktop/GraduateData/research1/run/mcpat/mcpat -infile '+path+'/'+i+' -print_level 1')
            f.write(i+'\n')
            f.write(content.read())



if __name__=='__main__':
    # split_stats("/home/lianghong/gem5/m5out")
    fill_template("/home/lianghong/gem5/m5out/mcpat_input","/home/lianghong/gem5/m5out/config.json","/home/lianghong/gem5/m5out/arm.xml")