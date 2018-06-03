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

def get_spearman_list(event_path, energy_path, time_window, perf_count):
    scalar=MinMaxScaler()
    event = power_m.get_event_all(event_path, time_window, perf_count)
    power = power_m.get_power_all(energy_path, time_window, perf_count)
    r = []
    length = len(event)
    for i in range(length):
        r.append((event[i][0], spearman(scalar.fit_transform(np.array([event[i][1]]).T), scalar.fit_transform(np.array([power[i][1]]).T)), spearman(scalar.fit_transform(np.array([event[i][1]]).T), scalar.fit_transform(np.array([power[i][2]]).T))))
    r.sort(key=lambda x:abs(x[1]),reverse=True)
    return r
