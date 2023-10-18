import numpy as np
from tqdm import *


def cal_weight(iter, func, class_num):
    cnt = np.zeros(class_num + 1)
    weights = []
    for sample in tqdm(iter, desc='calculate weight of samples PART 1/2'):
        res = func(sample['target'])
        res.append(0 if sum(res) else 1)
        cnt += np.array(res)

    for sample in tqdm(iter, desc='calculate weight of samples PART 2/2'):
        res = func(sample['target'])
        res.append(0 if sum(res) else 1)
        res = np.array(res) * cnt
        res = np.where(res > 0, res, 1e8)
        tmp = np.min(res)
        weights.append(1 / (tmp + 1) ** 0.5)

    with open('debug_check_weight.txt', 'w', encoding='utf-8') as O:
        print(cnt, file=O)
        print(weights, file=O)

    return weights
