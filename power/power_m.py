
def get_power_all(energy_path, time_window, perf_count, unit=2 ** 14):
    power = list()
    with open(energy_path, "r", encoding='utf-8') as f:
        line = f.readline()
        event_index = 1
        while line != '':
            pkg, pp0 = [], []
            count = 0
            while count < perf_count:
                power_list = line.split()
                pkg.append(int(power_list[0]) / unit)
                pp0.append(int(power_list[1]) / unit)
                count += 1
                line = f.readline()
            power.append([event_index, pkg, pp0])
            power.append([event_index + 1, pkg, pp0])
            power.append([event_index + 2, pkg, pp0])
            power.append([event_index + 3, pkg, pp0])
            event_index += 4

    power_len = len(power)
    # print(power_len)
    for i in range(0, power_len, 4):
        for j in range(1, perf_count):
            power[i][1][j - 1] = (power[i][1][j] - power[i][1][j - 1]) / time_window
            power[i][2][j - 1] = (power[i][2][j] - power[i][2][j - 1]) / time_window
        power[i][1].pop()
        power[i][2].pop()
    return power


def get_power_list(power_path, time_window, unit=2 ** 14):
    power = []
    with open(power_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        pkg, pp0 = [], []
        while line != '':
            data = line.split()
            pkg.append(int(data[0]) / unit)
            pp0.append(int(data[1]) / unit)
            line = f.readline()

        length = len(pkg)
        for i in range(length // 2):
            pkg[i] = (pkg[i * 2 + 1] - pkg[i * 2]) / time_window
            pp0[i] = (pp0[i * 2 + 1] - pp0[i * 2]) / time_window
        pkg = pkg[:length // 2]
        pp0 = pp0[:length // 2]
        for i in range(len(pkg) // 2):
            pkg[i] = (pkg[i * 2 + 1] + pkg[i * 2]) / 2
            pp0[i] = (pp0[i * 2 + 1] + pp0[i * 2]) / 2
        pp0 = pp0[:len(pp0) // 2]
        pkg = pkg[:len(pkg) // 2]
        power.append(pkg)
        power.append(pp0)
    return power


def get_event_all(event_path, time_window, perf_count):
    events = list()
    with open(event_path, "r", encoding='utf-8') as f:
        line = f.readline()
        event_index = 1
        while line != '':
            pmc0, pmc1, pmc2, pmc3 = [], [], [], []
            count = 0
            while count < perf_count:
                event_list = line.split()
                pmc0.append(int(event_list[0]))
                pmc1.append(int(event_list[1]))
                pmc2.append(int(event_list[2]))
                pmc3.append(int(event_list[3]))
                count += 1
                line = f.readline()
            events.append([event_index, pmc0])
            events.append([event_index + 1, pmc1])
            events.append([event_index + 2, pmc2])
            events.append([event_index + 3, pmc3])
            event_index += 4

    event_len = len(events)
    for i in range(event_len):
        for j in range(1, perf_count):
            events[i][1][j - 1] = (events[i][1][j] - events[i][1][j - 1]) / time_window
        events[i][1].pop()
    return events


def get_event_list(event_path, time_window):
    event_list = [list() for i in range(11)]
    with open(event_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != '':
            for i in range(2):
                data = line.split()
                event_list[0].append(int(data[0]))
                event_list[1].append(int(data[1]))
                event_list[2].append(int(data[2]))
                event_list[3].append(int(data[3]))
                event_list[8].append(int(data[4]))
                event_list[9].append(int(data[5]))
                event_list[10].append(int(data[6]))
                line = f.readline()
            for i in range(2):
                data = line.split()
                event_list[4].append(int(data[0]))
                event_list[5].append(int(data[1]))
                event_list[6].append(int(data[2]))
                event_list[7].append(int(data[3]))
                event_list[8].append(int(data[4]))
                event_list[9].append(int(data[5]))
                event_list[10].append(int(data[6]))
                line = f.readline()
        length = len(event_list[0])
        for j in range(11):
            for i in range(length // 2):
                event_list[j][i] = int((event_list[j][i * 2 + 1] - event_list[j][i * 2]) / time_window)
            event_list[j] = event_list[j][:length // 2]

    return event_list

