# -*- coding:utf-8 -*-
import math
import re
import json
import sys

import torch
import random
from tqdm import tqdm


def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十': hanzi = '一' + hanzi
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


def loads_str(data_str):
    cnt = 0
    this_w = []
    while True:
        cnt += 1
        try:
            result = json.loads(data_str)
            # print("最终json加载结果：{}".format(result))
            return result
        except Exception as e:
            # print("异常信息e：{}".format(e))
            error_index = re.findall(r"char (\d+)\)", str(e))
            if error_index:
                error_str = data_str[int(error_index[0])]
                this_w.append(data_str[int(error_index[0]) - 3:int(error_index[0]) + 3])
                print('带来错误的字符上下文:{}'.format(this_w), file=sys.stderr)
                data_str = data_str.replace(error_str, "")
                # print("替换异常字符串{} 后的文本内容{}".format(error_str, data_str))
                # 该处将处理结果继续递归处理
                # return loads_str(data_str)
            else:
                break
        if cnt >= 100:
            print(this_w)
            break


def get_elements_from_line(line):
    data = json.loads(line, strict=False)
    fact = data['fact'].replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '').replace(
        '<SIGN_LINE>', '')
    view = ''
    articles = [str(i) for i in sorted(data['laws'])]
    criminals_info = data['criminals_info']
    article_contents = articles
    relations = data['relations']
    return fact, articles, criminals_info, view, article_contents, relations


def split_fact_for_transform(fact):
    res = []
    for i in range(math.ceil(len(fact) / 512)):
        if fact[512 * i: 512 * (i + 1)]:
            res.append({'title': '段落{}'.format(i + 1), 'text': fact[512 * i: 512 * (i + 1)]})
        else:
            break
    return res


debug_output = open('./debug.txt', mode='w', encoding='utf-8')


def split_law_for_person(all_laws, criminals_info):
    marked_laws = set()
    all_laws = set([str(law) for law in all_laws])
    name_dict = {}
    for name in criminals_info:
        name_dict[name] = []
        states = criminals_info[name]['states']
        if 0 in states:
            states.remove(0)
        states = list(set(states))
        state_laws = set()
        for x in states:
            state_laws.add(str(get_state_law_mapper()[x]))
        if len(state_laws.intersection(all_laws)) != len(state_laws):
            # print('state wrong.')
            # print(sorted(state_laws))
            # print(sorted(all_laws))
            # return None
            pass
        name_dict[name].extend(list(state_laws.intersection(all_laws)))

        accus = criminals_info[name]['accusations']
        for accu in accus:
            if accu not in get_accusation_law_mapper().keys():
                continue
            law_ = get_accusation_law_mapper()[accu]
            if type(law_) == str:
                law_ = [law_]
            law_ = set([str(hanzi_to_num(x.replace('第', '').replace('条', ''))) for x in law_])
            if len(law_.intersection(all_laws)) < 1:
                print('accu wrong.')
                print(accu)
                print(sorted(law_))
                print(sorted(all_laws))
                # return None
            name_dict[name].extend(list(law_.intersection(all_laws)))

        for law in name_dict[name]:
            marked_laws.add(law)

    for name in criminals_info:
        for law in all_laws:
            if law not in marked_laws:
                name_dict[name].append(law)
        laws = name_dict[name]
        res = []
        for law in laws:
            res.append('<law_{}>'.format(law))
        name_dict[name] = ''.join(res)

    print(name_dict, file=debug_output)
    return name_dict


def get_str_from_relations(relations):
    dict_accu_relation = {}
    for relation in relations:
        if relation[0] not in dict_accu_relation:
            dict_accu_relation[relation[0]] = []
        dict_accu_relation[relation[0]].append(
            '[被告' + relation[1] + ']' + relation[2] + '了' + '[被告' + relation[3] + ']')
    relation_str_list = []
    for accu in dict_accu_relation:
        s = '，'.join(dict_accu_relation[accu])
        # s = '在{}中'.format(accu) + s
        relation_str_list.append(s)
    if relation_str_list:
        relation_str = '，'.join(relation_str_list)
    else:
        relation_str = '无'
    return relation_str


def get_str_from_states(criminals_info, name):
    states_set = set([str(x) for x in criminals_info[name]['states']])

    return ';'.join(list(states_set))


def prepare_complete_data_statistics(line, elements='1234'):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    facts = []
    # PROMPT 1 START
    facts.append(split_fact_for_transform(fact))
    relation_str = get_str_from_relations(relations)
    # PROMPT 1 END

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    name_law_dict = split_law_for_person(all_laws, criminals_info)

    info_list = []

    for name in criminals_info:
        relations_by_name = []
        for relation in relations:
            if '[被告{}]'.format(relation[1]) == name:
                relations_by_name.append(relation)
            if '[被告{}]'.format(relation[3]) == name:
                relations_by_name.append(relation)

        states_list = [str(x) for x in criminals_info[name]['states'] if int(x) != 0]

        info_list.append([len(re.findall('<', name_law_dict[name])), len(criminals_info[name]['accusations']),
                          len(relations_by_name), len(states_list)])

    return info_list


def prepare_complete_data(line, elements='1234', split=True):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    input_datas, targets = [], []
    facts = []
    # PROMPT 1 START
    input_datas.append('已知涉案被告代号是{}，那么被告之间的关系是？'.format(','.join(list(criminals_info.keys()))))
    facts.append(split_fact_for_transform(fact))
    relation_str = get_str_from_relations(relations)
    targets.append('他们之间的关系是{}'.format(relation_str))
    # PROMPT 1 END

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    name_law_dict = split_law_for_person(all_laws, criminals_info)

    for name in criminals_info:
        relations_by_name = []
        for relation in relations:
            if '[被告{}]'.format(relation[1]) == name:
                relations_by_name.append(relation)
            if '[被告{}]'.format(relation[3]) == name:
                relations_by_name.append(relation)
        relation_str_by_name = get_str_from_relations(relations_by_name)

        # PROMPT 2 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}。那么该被告存在什么量刑情节？'.format(name, relation_str_by_name))
        else:
            input_datas.append('着重考虑被告{}。该被告存在什么量刑情节？'.format(name))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告存在的量刑情节为{}。'.format(get_str_from_states(criminals_info, name)))
        # PROMPT 2 END

        # PROMPT 3 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告相关的法条和对应的罪名是？其刑期是？'.format(
                    name, relation_str_by_name, get_str_from_states(criminals_info, name)))
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(
                    name, get_str_from_states(criminals_info, name)))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告涉及的法条为{}，因此罪名为{}，刑期为{}。'.format(
                name_law_dict[name],
                ','.join(criminals_info[name]['accusations']),
                criminals_info[name]['term']
            )
        )
        # PROMPT 3 END

        # PROMPT 4 START
        if relation_str_by_name:
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告刑期应该是？对应的罪名是？相关的法条为？'.
                format(
                    name, relation_str_by_name, get_str_from_states(criminals_info, name)
                )
            )
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。其刑期应该是？对应的罪名是？相关的法条为？'.
                format(name, get_str_from_states(criminals_info, name))
            )
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告刑期为{}，因此罪名为{}，涉及的法条为{}。'.format(
                criminals_info[name]['term'],
                ','.join(criminals_info[name]['accusations']),
                name_law_dict[name]
            )
        )
        # PROMPT 4 END

    return input_datas, targets, facts


v2_set_for_accu_dict = {}


def prepare_complete_data_v2(line, elements='1234', split=True):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    accu_set = []
    for name in criminals_info:
        accu_set.extend(criminals_info[name]['accusations'])
    accu_set = set(accu_set)
    cnt = 0
    for accu in accu_set:
        if v2_set_for_accu_dict.get(accu, 0) > 400:
            cnt += 1
    # if cnt == len(accu_set):
    #     return None, None, None
    for accu in accu_set:
        v2_set_for_accu_dict[accu] = v2_set_for_accu_dict.get(accu, 0) + 1
    input_datas, targets = [], []
    facts = []
    # PROMPT 1 START
    input_datas.append('已知涉案被告代号是{}，那么被告之间的关系是？'.format(','.join(list(criminals_info.keys()))))
    facts.append(split_fact_for_transform(fact))
    relation_str = get_str_from_relations(relations)
    targets.append('他们之间的关系是{}'.format(relation_str))
    # PROMPT 1 END

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    name_law_dict = split_law_for_person(all_laws, criminals_info)

    for name in criminals_info:
        relations_by_name = []
        for relation in relations:
            if '[被告{}]'.format(relation[1]) == name:
                relations_by_name.append(relation)
            if '[被告{}]'.format(relation[3]) == name:
                relations_by_name.append(relation)
        relation_str_by_name = get_str_from_relations(relations_by_name)

        # PROMPT 2 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}。那么该被告存在什么量刑情节？'.format(name, relation_str_by_name))
        else:
            input_datas.append('着重考虑被告{}。该被告存在什么量刑情节？'.format(name))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告存在的量刑情节为{}。'.format(
                ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states'])))
        # PROMPT 2 END

        # PROMPT 3 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告相关的法条和对应的罪名是？其刑期是？'.format(
                    name, relation_str_by_name,
                    ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states'])))
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(
                    name, ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states'])))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告涉及的法条为{}，因此罪名为{}，刑期为{}。'.format(
                name_law_dict[name],
                ','.join(criminals_info[name]['accusations']),
                f'<term_{get_penalty_num(criminals_info[name]["term"])}>'
            )
        )
        # PROMPT 3 END

        # PROMPT 4 START
        if relation_str_by_name:
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告刑期应该是？对应的罪名是？相关的法条为？'.
                format(
                    name, relation_str_by_name,
                    ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states'])
                )
            )
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。其刑期应该是？对应的罪名是？相关的法条为？'.
                format(name, ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states']))
            )
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告刑期为{}，因此罪名为{}，涉及的法条为{}。'.format(
                f'<term_{get_penalty_num(criminals_info[name]["term"])}>',
                ','.join(criminals_info[name]['accusations']),
                name_law_dict[name]
            )
        )
        # PROMPT 4 END

    return input_datas, targets, facts


def prepare_complete_data_v3(line, elements='1234', split=True):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    accu_set = []
    for name in criminals_info:
        accu_set.extend(criminals_info[name]['accusations'])
    accu_set = set(accu_set)
    cnt = 0
    for accu in accu_set:
        if v2_set_for_accu_dict.get(accu, 0) > 400:
            cnt += 1
    # if cnt == len(accu_set):
    #     return None, None, None
    for accu in accu_set:
        v2_set_for_accu_dict[accu] = v2_set_for_accu_dict.get(accu, 0) + 1
    input_datas, targets = [], []
    facts = []
    # PROMPT 1 START
    input_datas.append('已知涉案被告代号是{}，那么被告之间的关系是？'.format(','.join(list(criminals_info.keys()))))
    facts.append(split_fact_for_transform(fact))
    relation_str = get_str_from_relations(relations)
    targets.append('他们之间的关系是{}'.format(relation_str))
    # PROMPT 1 END

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    name_law_dict = split_law_for_person(all_laws, criminals_info)

    for name in criminals_info:
        relations_by_name = []
        for relation in relations:
            if '[被告{}]'.format(relation[1]) == name:
                relations_by_name.append(relation)
            if '[被告{}]'.format(relation[3]) == name:
                relations_by_name.append(relation)
        relation_str_by_name = get_str_from_relations(relations_by_name)

        # PROMPT 2 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}。那么该被告存在什么量刑情节？'.format(name, relation_str_by_name))
        else:
            input_datas.append('着重考虑被告{}。该被告存在什么量刑情节？'.format(name))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告存在的量刑情节为{}。'.format(
                ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states'])))
        # PROMPT 2 END

        # PROMPT 3 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告相关的法条和对应的罪名是？其刑期是？'.format(
                    name, relation_str_by_name,
                    ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states'])))
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(
                    name, ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states'])))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告涉及的法条为{}，因此罪名为{}，刑期为{}。'.format(
                name_law_dict[name],
                ','.join(criminals_info[name]['accusations']),
                criminals_info[name]["term"]
            )
        )
        # PROMPT 3 END

        # PROMPT 4 START
        if relation_str_by_name:
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告刑期应该是？对应的罪名是？相关的法条为？'.
                format(
                    name, relation_str_by_name,
                    ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states'])
                )
            )
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。其刑期应该是？对应的罪名是？相关的法条为？'.
                format(name, ','.join(f'<state_{str(state)}>' for state in criminals_info[name]['states']))
            )
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告刑期为{}，因此罪名为{}，涉及的法条为{}。'.format(
                criminals_info[name]["term"],
                ','.join(criminals_info[name]['accusations']),
                name_law_dict[name]
            )
        )
        # PROMPT 4 END

    return input_datas, targets, facts


def prepare_case1_data(line):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    input_datas, targets = [], []
    facts = []

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    name_law_dict = split_law_for_person(all_laws, criminals_info)

    for name in criminals_info:
        # PROMPT 2 START
        input_datas.append('着重考虑被告{}。该被告存在什么量刑情节？'.format(name))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告存在的量刑情节为{}。'.format(get_str_from_states(criminals_info, name)))
        # PROMPT 2 END

        # PROMPT 3 START
        input_datas.append(
            '着重考虑被告{}具有的量刑情节为{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(
                name, get_str_from_states(criminals_info, name)))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告涉及的法条为{}，因此罪名为{}，刑期为{}。'.format(
                name_law_dict[name],
                ','.join(criminals_info[name]['accusations']),
                criminals_info[name]['term']
            )
        )
        # PROMPT 3 END

        # PROMPT 4 START
        input_datas.append(
            '着重考虑被告{}，具有的量刑情节为{}。其刑期应该是？对应的罪名是？相关的法条为？'.
            format(name, get_str_from_states(criminals_info, name))
        )
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告刑期为{}，因此罪名为{}，涉及的法条为{}。'.format(
                criminals_info[name]['term'],
                ','.join(criminals_info[name]['accusations']),
                name_law_dict[name]
            )
        )
        # PROMPT 4 END

    return input_datas, targets, facts


def prepare_case2_data(line):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    input_datas, targets = [], []
    facts = []
    # PROMPT 1 START
    input_datas.append('已知涉案被告代号是{}，那么被告之间的关系是？'.format(','.join(list(criminals_info.keys()))))
    facts.append(split_fact_for_transform(fact))
    relation_str = get_str_from_relations(relations)
    targets.append('他们之间的关系是{}'.format(relation_str))
    # PROMPT 1 END

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    name_law_dict = split_law_for_person(all_laws, criminals_info)

    for name in criminals_info:
        relations_by_name = []
        for relation in relations:
            if '[被告{}]'.format(relation[1]) == name:
                relations_by_name.append(relation)
            if '[被告{}]'.format(relation[3]) == name:
                relations_by_name.append(relation)
        relation_str_by_name = get_str_from_relations(relations_by_name)

        # PROMPT 3 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}。那么该被告相关的法条和对应的罪名是？其刑期是？'.
                format(name, relation_str_by_name)
            )
        else:
            input_datas.append(
                '着重考虑被告{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(name)
            )
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告涉及的法条为{}，因此罪名为{}，刑期为{}。'.format(
                name_law_dict[name],
                ','.join(criminals_info[name]['accusations']),
                criminals_info[name]['term']
            )
        )
        # PROMPT 3 END

        # PROMPT 4 START
        if relation_str_by_name:
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}。那么该被告刑期应该是？对应的罪名是？相关的法条为？'.
                format(
                    name, relation_str_by_name
                )
            )
        else:
            input_datas.append(
                '着重考虑被告{}。其刑期应该是？对应的罪名是？相关的法条为？'.
                format(name)
            )
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告刑期为{}，因此罪名为{}，涉及的法条为{}。'.format(
                criminals_info[name]['term'],
                ','.join(criminals_info[name]['accusations']),
                name_law_dict[name]
            )
        )
        # PROMPT 4 END

    return input_datas, targets, facts


def prepare_case3_data(line):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    input_datas, targets = [], []
    facts = []
    # PROMPT 1 START
    input_datas.append('已知涉案被告代号是{}，那么被告之间的关系是？'.format(','.join(list(criminals_info.keys()))))
    facts.append(split_fact_for_transform(fact))
    relation_str = get_str_from_relations(relations)
    targets.append('他们之间的关系是{}'.format(relation_str))
    # PROMPT 1 END

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    name_law_dict = split_law_for_person(all_laws, criminals_info)

    for name in criminals_info:
        relations_by_name = []
        for relation in relations:
            if '[被告{}]'.format(relation[1]) == name:
                relations_by_name.append(relation)
            if '[被告{}]'.format(relation[3]) == name:
                relations_by_name.append(relation)
        relation_str_by_name = get_str_from_relations(relations_by_name)

        # PROMPT 2 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}。那么该被告存在什么量刑情节？'.format(name, relation_str_by_name))
        else:
            input_datas.append('着重考虑被告{}。该被告存在什么量刑情节？'.format(name))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告存在的量刑情节为{}。'.format(get_str_from_states(criminals_info, name)))
        # PROMPT 2 END

        # PROMPT 3 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告相关的法条和对应的罪名是？其刑期是？'.format(
                    name, relation_str_by_name, get_str_from_states(criminals_info, name)))
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(
                    name, get_str_from_states(criminals_info, name)))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告涉及的法条为{}，因此罪名为{}，刑期为{}。'.format(
                name_law_dict[name],
                ','.join(criminals_info[name]['accusations']),
                criminals_info[name]['term']
            )
        )
        # PROMPT 3 END

    return input_datas, targets, facts


def prepare_case4_data(line):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    input_datas, targets = [], []
    facts = []
    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)
    name_law_dict = split_law_for_person(all_laws, criminals_info)

    for name in criminals_info:
        input_datas.append('预测{}对应的法条：'.format(name))
        facts.append(split_fact_for_transform(fact))
        targets.append('此被告涉及的法条为{}。'.format(name_law_dict[name]))
        input_datas.append('预测{}对应的罪名：'.format(name))
        facts.append(split_fact_for_transform(fact))
        targets.append('此被告罪名为{}。'.format(','.join(criminals_info[name]['accusations'])))
        input_datas.append('预测{}刑期：'.format(name))
        facts.append(split_fact_for_transform(fact))
        if criminals_info[name]['term'] == '':
            print(criminals_info[name]['term'])
        targets.append('此被告刑期为{}。'.format(criminals_info[name]['term']))

    return input_datas, targets, facts


def prepare_case5_data(line, split=True):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    input_datas, targets = [], []
    facts = []

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    if split:
        name_law_dict = split_law_for_person(all_laws, criminals_info)
    else:
        name_law_dict = {}
        for name in criminals_info:
            name_law_dict[name] = ''.join(['<law_{}>'.format(law) for law in articles])

    for name in criminals_info:
        # PROMPT 3 START
        input_datas.append(
            '着重考虑被告{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(name)
        )
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告涉及的法条为{}，因此罪名为{}，刑期为{}。'.format(
                name_law_dict[name],
                ','.join(criminals_info[name]['accusations']),
                criminals_info[name]['term']
            )
        )
        # PROMPT 3 END

        # PROMPT 4 START

        input_datas.append(
            '着重考虑被告{}。其刑期应该是？对应的罪名是？相关的法条为？'.
            format(name)
        )
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告刑期为{}，因此罪名为{}，涉及的法条为{}。'.format(
                criminals_info[name]['term'],
                ','.join(criminals_info[name]['accusations']),
                name_law_dict[name]
            )
        )
        # PROMPT 4 END

    return input_datas, targets, facts


def prepare_case6_data(line):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    input_datas, targets = [], []
    facts = []
    # PROMPT 1 START
    input_datas.append('已知涉案被告代号是{}，那么被告之间的关系是？'.format(','.join(list(criminals_info.keys()))))
    facts.append(split_fact_for_transform(fact))
    relation_str = get_str_from_relations(relations)
    targets.append('他们之间的关系是{}'.format(relation_str))
    # PROMPT 1 END

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    name_law_dict = split_law_for_person(all_laws, criminals_info)

    for name in criminals_info:
        relations_by_name = []
        for relation in relations:
            if '[被告{}]'.format(relation[1]) == name:
                relations_by_name.append(relation)
            if '[被告{}]'.format(relation[3]) == name:
                relations_by_name.append(relation)
        relation_str_by_name = get_str_from_relations(relations_by_name)

        # PROMPT 2 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}。那么该被告存在什么量刑情节？'.format(name, relation_str_by_name))
        else:
            input_datas.append('着重考虑被告{}。该被告存在什么量刑情节？'.format(name))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告存在的量刑情节为{}。'.format(get_str_from_states(criminals_info, name)))
        # PROMPT 2 END

        # PROMPT 4 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告刑期应该是？对应的罪名是？相关的法条为？'.
                format(
                    name, relation_str_by_name, get_str_from_states(criminals_info, name)
                )
            )
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。其刑期应该是？对应的罪名是？相关的法条为？'.
                format(name, get_str_from_states(criminals_info, name))
            )
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告刑期为{}，因此罪名为{}，涉及的法条为{}。'.format(
                criminals_info[name]['term'],
                ','.join(criminals_info[name]['accusations']),
                name_law_dict[name]
            )
        )
        # PROMPT 4 END

    return input_datas, targets, facts


def prepare_case7_data(line):
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
    input_datas, targets = [], []
    facts = []
    # PROMPT 1 START
    input_datas.append('已知涉案被告代号是{}，那么被告之间的关系是？'.format(','.join(list(criminals_info.keys()))))
    facts.append(split_fact_for_transform(fact))
    relation_str = get_str_from_relations(relations)
    targets.append('他们之间的关系是{}'.format(relation_str))
    # PROMPT 1 END

    all_laws = []
    for key in sorted(get_all_articles().keys()):
        if key in articles:
            all_laws.append(key)

    name_law_dict = split_law_for_person(all_laws, criminals_info)

    for name in criminals_info:
        relations_by_name = []
        for relation in relations:
            if '[被告{}]'.format(relation[1]) == name:
                relations_by_name.append(relation)
            if '[被告{}]'.format(relation[3]) == name:
                relations_by_name.append(relation)
        relation_str_by_name = get_str_from_relations(relations_by_name)

        # PROMPT 2 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}。那么该被告存在什么量刑情节？'.format(name, relation_str_by_name))
        else:
            input_datas.append('着重考虑被告{}。该被告存在什么量刑情节？'.format(name))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告存在的量刑情节为{}。'.format(get_str_from_states(criminals_info, name)))
        # PROMPT 2 END

        # PROMPT 3.1 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告相关的法条是？'.format(
                    name, relation_str_by_name, get_str_from_states(criminals_info, name)))
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。该被告相关的法条是？'.format(
                    name, get_str_from_states(criminals_info, name)))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告涉及的法条为{}。'.format(
                name_law_dict[name]
            )
        )
        # PROMPT 3.1 END

        # PROMPT 3.2 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告的罪名是？'.format(
                    name, relation_str_by_name, get_str_from_states(criminals_info, name)))
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。该被告的罪名是？'.format(
                    name, get_str_from_states(criminals_info, name)))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告罪名为{}。'.format(
                ','.join(criminals_info[name]['accusations'])
            )
        )
        # PROMPT 3.2 END

        # PROMPT 3.3 START
        if relation_str_by_name != '无':
            input_datas.append(
                '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告刑期是？'.format(
                    name, relation_str_by_name, get_str_from_states(criminals_info, name)))
        else:
            input_datas.append(
                '着重考虑被告{}，具有的量刑情节为{}。该被告刑期是？'.format(
                    name, get_str_from_states(criminals_info, name)))
        facts.append(split_fact_for_transform(fact))
        targets.append(
            '此被告刑期为{}。'.format(
                criminals_info[name]['term']
            )
        )
        # PROMPT 3.3 END

    return input_datas, targets, facts


def get_article_idx(article):
    p = re.search(r'第?([0-9]{1,3})条?', article)
    if p:
        article = p.group(1)
    article2idx = get_all_articles()
    if article in article2idx:
        return article2idx[article]
    else:
        print('出错的法律{}'.format(article), file=sys.stderr)
        return 0


def get_all_articles():
    return {'67': 0, '72': 1, '293': 2, '25': 3, '26': 4, '65': 5, '73': 6, '264': 7, '52': 8, '53': 9, '27': 10,
            '64': 11, '347': 12, '348': 13, '234': 14, '232': 15, '69': 16, '68': 17, '292': 18, '77': 19, '303': 20,
            '266': 21, '23': 22, '4': 23, '356': 24, '56': 25, '71': 26, '86': 27, '55': 28, '61': 29, '70': 30,
            '238': 31, '59': 32, '57': 33, '2': 34, '1': 35, '47': 36, '45': 37, '48': 38, '38': 39, '41': 40, '76': 41,
            '62': 42, '8': 43, '44': 44, '42': 45, '349': 46, '3': 47, '6': 48, '75': 49, '36': 50, '312': 51,
            '224': 52, '63': 53, '19': 54, '390': 55, '383': 56, '7': 57, '382': 58, '389': 59, '5': 60, '277': 61,
            '11': 62, '357': 63, '12': 64, '17': 65, '9': 66, '22': 67, '198': 68, '50': 69, '133': 70, '275': 71,
            '29': 72, '74': 73, '51': 74, '196': 75, '385': 76, '397': 77, '386': 78, '155': 79, '37': 80, '14': 81,
            '263': 82, '274': 83, '128': 84, '279': 85, '345': 86, '365': 87, '20': 88, '24': 89, '93': 90, '15': 91,
            '302': 92, '54': 93, '58': 94, '97': 95, '157': 96, '31': 97, '81': 98, '203': 99, '13': 100, '125': 101,
            '344': 102, '354': 103, '384': 104, '18': 105, '310': 106, '87': 107, '176': 108, '172': 109, '241': 110,
            '334': 111, '280': 112, '290': 113, '30': 114, '39': 115, '40': 116, '393': 117, '60': 118, '43': 119,
            '34': 120, '16': 121, '201': 122, '388': 123, '193': 124, '88': 125, '226': 126, '28': 127, '307': 128,
            '83': 129, '21': 130, '269': 131}


def get_accusation_mapper():
    return {'诈骗罪': 0, '合同诈骗罪': 1, '保险诈骗罪': 2, '贷款诈骗罪': 3, '招摇撞骗罪': 4, '盗窃罪': 5,
            '盗伐林木罪': 6, '故意伤害罪': 7,
            '寻衅滋事罪': 8, '聚众斗殴罪': 9, '故意杀人罪': 10, '赌博罪': 11, '开设赌场罪': 12, '受贿罪': 13,
            '行贿罪': 14, '贪污罪': 15,
            '妨害公务罪': 16, '非法拘禁罪': 17, '敲诈勒索罪': 18, '贩卖毒品罪': 19, '贩卖、运输毒品罪': 19,
            '运输毒品罪': 19, '制造毒品罪': 19,
            '贩卖、制造毒品罪': 19, '走私、贩卖毒品罪': 19, '走私、贩卖、运输毒品罪': 19, '走私、运输毒品罪': 19,
            '走私毒品罪': 19, '贩卖、运输、制造毒品罪': 19,
            '走私、贩卖、运输、制造毒品罪': 19, '制造、贩卖毒品罪': 19, '非法持有毒品罪': 20, '窝藏毒品罪': 21,
            '窝藏、转移毒品罪': 21, '转移毒品罪': 21,
            '包庇毒品犯罪分子罪': 22}


def get_accusation_idx(accu):
    accusation2idx = get_accusation_mapper()
    if accu in accusation2idx:
        return accusation2idx[accu]
    else:
        return None


def get_accusation_law_mapper():
    return {'诈骗罪': '第二百六十六条', '合同诈骗罪': '第二百二十四条', '保险诈骗罪': '第一百九十八条',
            '贷款诈骗罪': '第一百九十三条', '招摇撞骗罪': '第二百七十九条',
            '盗窃罪': '第二百六十四条', '盗伐林木罪': '第三百四十五条', '故意伤害罪': '第二百三十四条',
            '寻衅滋事罪': '第二百九十三条',
            '聚众斗殴罪': '第二百九十二条', '故意杀人罪': '第二百三十二条', '赌博罪': '第三百零三条',
            '开设赌场罪': '第三百零三条', '受贿罪': '第三百八十五条',
            '行贿罪': '第三百八十九条', '贪污罪': ['第三百八十二条', '第三百八十三条'],
            '妨害公务罪': ['第二百四十二条', '第二百七十七条'], '非法拘禁罪': '第二百三十八条',
            '敲诈勒索罪': '第二百七十四条',
            '贩卖毒品罪': '第三百四十七条', '贩卖、运输毒品罪': '第三百四十七条', '运输毒品罪': '第三百四十七条',
            '制造毒品罪': '第三百四十七条',
            '贩卖、制造毒品罪': '第三百四十七条', '走私、贩卖毒品罪': '第三百四十七条',
            '走私、贩卖、运输毒品罪': '第三百四十七条', '走私、运输毒品罪': '第三百四十七条',
            '走私毒品罪': '第三百四十七条',
            '贩卖、运输、制造毒品罪': '第三百四十七条',
            '走私、贩卖、运输、制造毒品罪': '第三百四十七条', '制造、贩卖毒品罪': '第三百四十七条',
            '非法持有毒品罪': '第三百四十八条', '窝藏毒品罪': '第三百四十九条', '窝藏、转移毒品罪': '第三百四十九条',
            '转移毒品罪': '第三百四十九条',
            '包庇毒品犯罪分子罪': '第三百四十九条'}


def get_state_law_mapper():
    return {
        1: 17, 2: 19, 3: 27, 4: 23, 5: 67, 6: 67, 7: 68, 8: 65
    }


# def get_accusation_law(accu):
#     if type(accu) == int:


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


def get_penalty_num(penalty):
    if '死刑' in penalty:
        return 0
    elif '无期徒刑' in penalty:
        return 0
    else:
        p = re.search(
            r'(有期徒刑|拘役|管制)(([一二三四五六七八九十零]{1,3})个?年)?(([一二三四五六七八九十零]{1,3})个?月)?',
            penalty)
        if not p:
            print('出错的刑期 {}'.format(penalty), file=sys.stderr)
            return 0
        else:
            if p.group(3):
                year = int(hanzi_to_num(p.group(3)))
            else:
                year = 0
            if p.group(5):
                month = int(hanzi_to_num(p.group(5)))
            else:
                month = 0
            num = year * 12 + month
            return get_class_of_month(num)


def get_penalty_num_month(penalty):
    if '死刑' in penalty:
        return 0
    elif '无期徒刑' in penalty:
        return 0
    else:
        p = re.search(
            r'(有期徒刑|拘役|管制)(([一二三四五六七八九十零]{1,3})个?年)?(([一二三四五六七八九十零]{1,3})个?月)?',
            penalty)
        if not p:
            print('出错的刑期 {}'.format(penalty), file=sys.stderr)
            return 0
        else:
            if p.group(3):
                year = int(hanzi_to_num(p.group(3)))
            else:
                year = 0
            if p.group(5):
                month = int(hanzi_to_num(p.group(5)))
            else:
                month = 0
            num = year * 12 + month
            return num


def get_statistics():
    files = ['./data_train.json', './data_valid.json', './data_test.json']
    data_list = []
    for file_name in files:
        I = open(file_name, mode='r', encoding='utf-8')
        data_list.extend(I.readlines())

    fact_length_list = []
    name_num_list = []
    info_list = []

    for line in data_list:
        fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
        fact_length_list.append(len(fact))
        name_num_list.append(len(criminals_info))
        info_list.extend(prepare_complete_data_statistics(line))
    import numpy as np
    info_list = np.array(info_list)

    print('fact max, mean len:', np.max(fact_length_list), np.mean(fact_length_list))
    print('criminals sum num, mean num:', np.sum(name_num_list), np.mean(name_num_list))
    print('law mean num:', np.mean(info_list[:, 0]))
    print('accu mean num:', np.mean(info_list[:, 1]))
    print('relation mean num:', np.mean(info_list[:, 2]))
    print('states mean num', np.mean(info_list[:, 3]))


def get_statistics_for_daoqie():
    files = ['./data_train.json', './data_valid.json', './data_test.json']
    data_list = []
    for file_name in files:
        I = open(file_name, mode='r', encoding='utf-8')
        data_list.extend(I.readlines())

    fact_length_list = []
    name_num_list = []
    info_list = []

    term_list = []
    for line in data_list:
        fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(line)
        for name in criminals_info:
            accu_law_states_list = criminals_info[name]['accu_law_states']
            flag = False
            for tt in accu_law_states_list:
                if tt[0] == '寻衅滋事罪':
                    flag = True
            if flag:
                term = get_penalty_num_month(criminals_info[name]['term'])
                term_list.append(term)

    print(len(term_list))
    print(sum(term_list) / len(term_list))


if __name__ == "__main__":
    # get_statistics_for_daoqie()
    pass
