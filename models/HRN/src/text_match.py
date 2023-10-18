import re
from copy import deepcopy

tasks = ['laws', 'accus', 'term']
blank_task = {
    'laws': [0] * 22,
    'accus': [0] * 23,
    'term': -1,
    # 'states': [0] * 8,
}


def re_match(text):
    assert isinstance(text, str)
    text = text.replace('[', '').replace(']', '')
    res = {
        'laws': get_laws(text),
        'accus': get_accus(text),
        'term': get_penalty_num(text),
    }
    return res


def get_laws(text):
    res = deepcopy(blank_task['laws'])
    for r in re.finditer(r'<law_([0-9]{1,3})>', text):
        try:
            res[ALL_LAWS[r.group(1)]] = 1
        except KeyError:
            pass
    return res


def get_accus(text):
    res = deepcopy(blank_task['accus'])
    try:
        accus = re.search(r'因此罪名为(.+?),刑期为', text).group(1)
        accus = accus.split(',')
        for accu in accus:
            if accu in ALL_ACCUS:
                res[ALL_ACCUS[accu]] = 1
        return res
    except AttributeError:
        past_accus = ''
        for accu, idx in ALL_ACCUS.items():
            if accu in text and accu not in past_accus:
                past_accus += accu
                res[idx] = 1
        return res


def get_penalty_num(penalty):
    if '死刑' in penalty or '无期徒刑' in penalty:
        return 0
    else:
        p = re.search(
            r'(有期徒刑|拘役|管制)(([一二三四五六七八九十零]{1,3})个?年)?(([一二三四五六七八九十零]{1,3})个?月)?',
            penalty
        )
        if not p:
            return -1
        else:
            year = int(hanzi_to_num(p.group(3))) if p.group(3) else 0
            month = int(hanzi_to_num(p.group(5))) if p.group(5) else 0
            num = year * 12 + month
            return get_class_of_month(num)


def get_class_of_month(num):
    if num > 20 * 12:
        num = 0
    elif num > 10 * 12:
        num = 1
    elif num > 7 * 12:
        num = 2
    elif num > 5 * 12:
        num = 3
    elif num > 3 * 12:
        num = 4
    elif num > 2 * 12:
        num = 5
    elif num > 1 * 12:
        num = 6
    elif num > 9:
        num = 7
    elif num > 6:
        num = 8
    elif num > 0:
        num = 9
    else:
        num = 10
    return num


def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
         '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十':
        hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)


def add_tokenizer_tokens(tokenizer, add_state_token=True, add_beigao_token=True):
    tokenizer.add_tokens(
        list(reversed(['<law_{}>'.format(key) for key in sorted(ALL_LAWS)])))
    if add_state_token:
        tokenizer.add_tokens(
            list(reversed(['<state_{}>'.format(key) for key in range(9)])))
    if add_beigao_token:
        alphabet = [''] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        all_beigao_names = []
        for a in alphabet:
            for b in alphabet:
                if b == '':
                    continue
                all_beigao_names.append(f'[被告{a + b}]')
        tokenizer.add_tokens(all_beigao_names)
    return tokenizer


ALL_LAWS = {'347': 0, '348': 1, '266': 2, '293': 3, '303': 4, '264': 5, '234': 6, '389': 7, '292': 8, '224': 9,
            '238': 10, '277': 11, '198': 12, '349': 13, '382': 14, '385': 15, '279': 16, '388': 17, '232': 18,
            '345': 19, '274': 20, '193': 21}
ALL_ACCUS = {'诈骗罪': 0, '合同诈骗罪': 1, '保险诈骗罪': 2, '贷款诈骗罪': 3, '招摇撞骗罪': 4, '盗窃罪': 5,
             '盗伐林木罪': 6, '故意伤害罪': 7, '寻衅滋事罪': 8, '聚众斗殴罪': 9, '故意杀人罪': 10, '赌博罪': 11,
             '开设赌场罪': 12, '受贿罪': 13, '行贿罪': 14, '贪污罪': 15, '妨害公务罪': 16, '非法拘禁罪': 17,
             '敲诈勒索罪': 18, '贩卖毒品罪': 19, '贩卖、运输毒品罪': 19, '运输毒品罪': 19, '制造毒品罪': 19,
             '贩卖、制造毒品罪': 19, '走私、贩卖毒品罪': 19, '走私、贩卖、运输毒品罪': 19, '走私、运输毒品罪': 19,
             '走私毒品罪': 19, '贩卖、运输、制造毒品罪': 19, '走私、贩卖、运输、制造毒品罪': 19, '制造、贩卖毒品罪': 19,
             '非法持有毒品罪': 20, '窝藏毒品罪': 21, '窝藏、转移毒品罪': 21, '转移毒品罪': 21, '包庇毒品犯罪分子罪': 22}
_ = sorted(ALL_ACCUS, key=lambda x: len(x), reverse=True)
ALL_ACCUS = {k: ALL_ACCUS[k] for k in _}
with open('./tmp/debug_all_accu.txt', 'w', encoding='utf-8') as O:
    print(ALL_ACCUS, file=O)

ALL_STATES = {
    "年龄大于75岁": 1,
    "聋哑或盲人": 2,
    "从犯": 3,
    "犯罪未遂": 4,
    "自首": 5,
    "坦白": 6,
    "立功": 7,
    "累犯": 8,
    "其他": 0
}

INV_STATES = {v: k for k, v in ALL_STATES.items()}
