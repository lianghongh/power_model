from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
import numpy as np

def get_temp_power(temp_path, time_window, unit=2 ** 14):
    temp = []
    with open(temp_path, "r", encoding='utf-8') as f:
        line = f.readline()
        pkg_temp, core0, core1, core2, core3, pkg, pp0 = [], [], [], [], [], [], []
        while line != '':
            temp_list = line.split()
            pkg_temp.append(float(temp_list[0]))
            core0.append(float(temp_list[1]))
            core1.append(float(temp_list[2]))
            core2.append(float(temp_list[3]))
            core3.append(float(temp_list[4]))
            pkg.append(int(temp_list[5]) / unit)
            pp0.append(int(temp_list[6]) / unit)
            line = f.readline()

        length = len(pkg_temp)
        for i in range(1, length):
            pkg_temp[i - 1] = (pkg_temp[i] + pkg_temp[i - 1]) / 2
            core0[i - 1] = (core0[i] + core0[i - 1]) / 2
            core1[i - 1] = (core1[i] + core1[i - 1]) / 2
            core2[i - 1] = (core2[i] + core2[i - 1]) / 2
            core3[i - 1] = (core3[i] + core3[i - 1]) / 2
            pkg[i - 1] = (pkg[i] - pkg[i - 1]) / time_window
            pp0[i - 1] = (pp0[i] - pp0[i - 1]) / time_window
        pkg_temp.pop()
        core0.pop()
        core1.pop()
        core2.pop()
        core3.pop()
        pkg.pop()
        pp0.pop()
        temp.append(pkg_temp)
        temp.append(core0)
        temp.append(core1)
        temp.append(core2)
        temp.append(core3)
        temp.append(pkg)
        temp.append(pp0)

    return temp

def Temperature_LinearRegression(temp_path,time_window,unit=2**14):
    temp=get_temp_power(temp_path,time_window,unit)
    temp_data=np.array(temp[0]).reshape(-1,1)
    power_data=np.array(temp[5]).reshape(-1,1)
    temp_train,temp_test,power_train,power_test=train_test_split(temp_data,power_data,train_size=0.95)
    model=LinearRegression()
    model.fit(temp_train,power_train)
    plt.plot(temp_data,power_data,'o',color='g')
    plt.plot(temp_train,model.predict(temp_train),color='r')
    plt.show()
    return model.coef_,model.intercept_