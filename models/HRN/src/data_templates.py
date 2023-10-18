import random

templates = [
    {
        "in": "你需要给出{name}在此案中涉及的，法条;罪名;刑期。[]中为答案。",
        "out": "法条: [{laws}]; 罪名: [{accus}]; 刑期: [{term}];",
        "stage": 3
    },
    {
        "in": "你需要给出{name}在此案中涉及的，刑期;罪名;法条。[]中为答案。",
        "out": "刑期: [{term}]; 罪名: [{accus}]; 法条: [{laws}];",
        "stage": 4
    },
    {
        "in": "已知{name}在此案中存在的量刑情节为{states}，你需要给出{name}在此案中涉及的，刑期;罪名;法条。[]中为答案。",
        "out": "刑期: [{term}]; 罪名: [{accus}]; 法条: [{laws}];",
        "stage": 4
    },
    {
        "in": "已知{name}在此案中存在的量刑情节为{states}，你需要给出{name}在此案中涉及的，法条;罪名;刑期。[]中为答案。",
        "out": "法条: [{laws}]; 罪名: [{accus}]; 刑期: [{term}];",
        "stage": 3
    },
    {
        "in": "已知{name}在此案中存在的量刑情节为{states}和关系为{relations}，你需要给出{name}在此案中涉及的，法条;罪名;刑期。[]中为答案。",
        "out": "法条: [{laws}]; 罪名: [{accus}]; 刑期: [{term}];",
        "stage": 3
    },
    {
        "in": "已知{name}在此案中存在的关系为{relations}，量刑情节为{states}，你需要给出{name}在此案中涉及的，法条;罪名;刑期。[]中为答案。",
        "out": "法条: [{laws}]; 罪名: [{accus}]; 刑期: [{term}];",
        "stage": 3
    },
    {
        "in": "你需要给出{name}在此案中与其他被告存在的关系，列举其存在的关系和量刑情节。[]中为答案。",
        "out": "关系: [{relations}]; 情节: [{states}];",
        "stage": 2
    },
    {
        "in": "判案，{name}",
        "out": "法条: [{laws}]; 罪名: [{accus}]; 刑期: [{term}];",
        "stage": 3
    },
    {
        "in": "判案，{name}, {relations}, {states}",
        "out": "法条: [{laws}]; 罪名: [{accus}]; 刑期: [{term}];",
        "stage": 3
    },
    {
        "in": "判案，{name}, {relations}, {states}, simple mode",
        "out": "[{laws}]; [{accus}]; [{term}];",
        "stage": 3
    },
    {
        "in": "已知涉案被告代号是{name}，那么被告之间的关系是？",
        "out": "他们之间的关系是{relations}",
        "stage": 1
    },
    {
        "in": "已知被告{name}涉及到的犯罪关系为{relations}。那么该被告存在什么量刑情节？",
        "out": "此被告存在的量刑情节为{states}",
        "stage": 2
    },
    {
        "in": "已知被告{name}涉及到的犯罪关系为{relations}，具有的量刑情节为{states}。那么该被告相关的法条和对应的罪名是？其刑期是？[]中为答案。",
        "out": "此被告涉及的法条为[{laws}]，因此罪名为[{accus}]，刑期为[{term}]",
        "stage": 3
    },
    {
        "in": "已知被告{name}涉及到的犯罪关系为{relations}，具有的量刑情节为{states}。那么该被告相关的法条和对应的罪名是？其刑期是？",
        "out": "此被告涉及的法条为{laws}，因此罪名为{accus}，刑期为{term}",
        "stage": 3
    },
    {
        "in": "已知被告{name}涉及到的犯罪关系为{relations}，具有的量刑情节为{states}。那么该被告刑期应该是？对应的罪名是？相关的法条为？",
        "out": "此被告刑期为{term}，因此罪名为{accus}，涉及的法条为{laws}。",
        "stage": 4
    },
]

stage1 = {
    "in": "已知涉案被告代号是{name}，那么被告之间的关系是？",
    "out": "他们之间的关系是{relations}",
    "stage": 1
}
stage2 = {
    "in": "已知被告{name}涉及到的犯罪关系为{relations}。那么该被告存在什么量刑情节？",
    "out": "此被告存在的量刑情节为{states}",
    "stage": 2
}
stage3 = {
    "in": "已知被告{name}涉及到的犯罪关系为{relations}，具有的量刑情节为{states}。那么该被告相关的法条和对应的罪名是？其刑期是？[]中为答案。",
    "out": "此被告涉及的法条为[{laws}]，因此罪名为[{accus}]，刑期为[{term}]",
    "stage": 3,
}
stage3r = {
    "in": "已知被告{name}涉及到的犯罪关系为{relations}，具有的量刑情节为{states}。那么该被告刑期应该是？对应的罪名是？相关的法条为？",
    "out": "此被告刑期为{term}，因此罪名为{accus}，涉及的法条为{laws}。",
    "stage": 4
}
stage_af = {
    "in": "对被告{name}进行判案([]中为答案，以法条，罪名，刑期的顺序输出判决结果)：",
    "out": "被告{name}涉及到的犯罪关系为{relations}，具有的量刑情节为{states}。"
           "因此此被告涉及的法条为[{laws}]，因此罪名为[{accus}]，刑期为[{term}]。",
    "stage": 1
}
stage_ar = {
    "in": "对被告{name}进行判案([]中为答案，以刑期，罪名，法条的顺序输出判决结果)：",
    "out": "被告{name}涉及到的犯罪关系为{relations}，具有的量刑情节为{states}。"
           "因此此被告刑期为[{term}]，因此罪名为[{accus}]，涉及的法条为[{laws}]。",
    "stage": -1
}
strict_stages = [stage1, stage2, stage3, stage3r]
all_in_two = [stage_af, stage_ar]
only_final = [templates[0].copy(), templates[1].copy()]


def get_related_relations(relations, person):
    t_dic = {}
    for relation in relations:
        relation[1] = f'[被告{relation[1]}]'
        relation[3] = f'[被告{relation[3]}]'
        if relation[1] == person:
            if relation[2] in t_dic:
                t_dic[relation[2]].append(relation[3])
            else:
                t_dic[relation[2]] = [relation[3]]

    res = ''
    for key in t_dic:
        res += f'{person}{key}{"|".join(t_dic[key])};'
    return res


idx_stage = None
idx = None
template_ex_x = {}


def init_templates(exclude_stage):
    n_ts = []
    for template in templates:
        if template['stage'] != exclude_stage:
            n_ts.append(template)
    return n_ts


for i in range(1, 5):
    template_ex_x[i] = init_templates(i)


def apply_template(sample, seed=42, k=2, test=False, exclude_stage=None):
    global idx, idx_stage

    results = []

    if isinstance(test, int):
        if idx_stage is None or idx_stage != test:
            idx_stage = test
            idx = 0

    here_templates = templates
    if exclude_stage is not None and isinstance(exclude_stage, int):
        here_templates = template_ex_x[exclude_stage]

    for name in sample['criminals_info']:
        cif = sample['criminals_info'][name]
        if '其他' in cif['raw_states']:
            cif['raw_states'].remove('其他')
        states = ','.join(cif['raw_states']) if (cif['raw_states'] is not None and len(cif['raw_states']) > 0) else [
            '无']
        laws = ','.join([f'<law_{law}>' for law in cif['laws']])
        accus = ','.join(cif['accusations'])
        term = cif['term']
        _ = get_related_relations(sample['relations'], name)
        relations = _ if (_ is not None and len(_) > 0) else '无'

        if isinstance(test, bool) and not test:
            __ = random.sample(here_templates, k=k)
        elif isinstance(test, int):
            __ = [strict_stages[test]]
        elif isinstance(test, str) and test == 'all_in_two':
            __ = all_in_two
        elif isinstance(test, str) and test == 'only_final':
            __ = only_final
        else:
            raise ValueError

        if exclude_stage == 1:
            relations = '无'
        elif exclude_stage == 2:
            states = '无'
        elif exclude_stage == 3:
            pass
        elif exclude_stage == 4:
            pass
        elif exclude_stage == 'only_final':
            __ = only_final
        elif exclude_stage is None:
            pass
        else:
            raise ValueError('No this option!')

        for template in __:
            _ = {
                'question': template['in'].format(name=name, relations=relations, states=states),
                'inputs': '案件fact: ' + sample['fact'],
                'target': template['out'].format(
                    relations=relations, states=states, laws=laws, accus=accus, term=term, name=name
                ),
                'stage': template['stage']
            }
            if isinstance(test, int):
                _['name'] = name
                _['idx'] = idx
                idx += 1
            results.append(_)

    return results
