import re

import requests.exceptions
from datasets import load_dataset
import sys
import os
from dataclasses import dataclass, field
from transformers import (HfArgumentParser, TrainingArguments, set_seed, AutoTokenizer, Trainer,
                          EvalPrediction, MT5Config, MT5ForConditionalGeneration, DataCollatorWithPadding,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from src.model import FiDMT5
from src.text_match import re_match, tasks, add_tokenizer_tokens, ALL_LAWS, ALL_ACCUS, get_class_of_month
from sklearn import metrics
import numpy as np
from torch import nn
from typing import Dict, Union, Any, Optional, List
import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from src.cal_sample_weight import cal_weight
import src


@dataclass
class MyTrainingArguments:
    data_version: str = field(
        default='single'
    )
    model_name: str = field(
        default='google/mt5-base'
    )
    context_num: int = field(
        default=4
    )
    per_context_len: int = field(
        default=512
    )
    use_weight: bool = field(
        default=False
    )
    weight_train_num: int = field(
        default=40000
    )
    valid_num: int = field(
        default=30000
    )


def get_metrics(p, t):
    return {
        'acc': metrics.accuracy_score(t, p),
        'p': metrics.precision_score(t, p, average='macro'),
        'r': metrics.recall_score(t, p, average='macro'),
        'f1': metrics.f1_score(t, p, average='macro')
    }


def task_metric(p, tokenizer, data) -> dict:
    preds = p
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    p_res, t_res = {'laws': [], 'accus': [], 'term': []}, {'laws': [], 'accus': [], 'term': []}
    debug_output = open('debug_single_all.txt', 'w', encoding='utf-8')
    for p, sample in zip(preds, data):
        accu = sample['raw_accu']
        law = sample['raw_law']
        term = sample['time']
        p_dict = re_match(p)
        t_dict = re_match(f"{accu} <law_{law}>")
        t_dict['term'] = get_class_of_month(term)
        print(p, file=debug_output)
        print(p_dict, file=debug_output)
        print(t_dict, file=debug_output)
        print('-' * 80, file=debug_output)
        for task in tasks:
            p_res[task].append(p_dict[task])
            t_res[task].append(t_dict[task])
    debug_output.close()
    res = {}
    acc_sum = 0
    f1_sum = 0
    for task in tasks:
        metric = get_metrics(p_res[task], t_res[task])
        for k, v in metric.items():
            if k == 'f1':
                f1_sum += v
            if k == 'acc':
                acc_sum += v
            res[f'{task}-{k}'] = v
    res['acc_sum'] = acc_sum
    res['f1_sum'] = f1_sum
    res['acc_f1_sum'] = acc_sum + f1_sum
    return res


def main():
    parser = HfArgumentParser((MyTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        my_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        my_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    data_fields = {
        'test': f'../../dataset/dataset-single/test_cs_v1.03.jsonl',
    }

    tokenizer = AutoTokenizer.from_pretrained(my_args.model_name)
    tokenizer = add_tokenizer_tokens(tokenizer)

    model_cfg = MT5Config.from_pretrained(my_args.model_name)
    setattr(model_cfg, 'n_passages', my_args.context_num)
    setattr(model_cfg, 'passage_len', my_args.per_context_len)
    model = FiDMT5(model_cfg).cpu()
    model.unwrap_encoder()
    model.resize_token_embeddings(250100 + 22 + 11 + 9 + 27 * 26)
    model.wrap_encoder()
    model.load_state_dict(torch.load(os.path.join(training_args.resume_from_checkpoint, "pytorch_model.bin")))
    model = model.cuda()

    def fact_pre(sample):
        return {
            'inputs': ("你需要给出{name}在此案中涉及的，法条;罪名;刑期。[]中为答案。".format(
                name='[被告A]', relations='无', states='无')
                      + sample['fact_cut'].replace(' ', '').replace('被告人', '[被告A]'))[:768],
            'raw_accu': sample['raw_accu'] + '罪'
        }

    def maintain_in_domain(sample):
        if sample['raw_accu'] in ALL_ACCUS and sample['raw_law'] in ALL_LAWS:
            return True
        else:
            return False

    def tokenize_input(sample):
        res = tokenizer(
            sample['inputs'],
            max_length=my_args.context_num * my_args.per_context_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return res

    data = load_dataset("json", data_files=data_fields)
    data = data.map(fact_pre, batched=False, num_proc=16)
    data = data.filter(maintain_in_domain, batched=False, num_proc=16)
    with open('debug_preprocess.txt', 'w', encoding='utf-8') as O:
        print(data['test'][:10], file=O)
    data = data.map(tokenize_input, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=None,
    )

    data['test'] = data['test'].shuffle()
    res = trainer.predict(data['test'])
    torch.save([res, tokenizer, data['test']], './tmp/single_res.pkl')


def just_task():
    res, tokenizer, data_test = torch.load('./tmp/single_res.pkl')
    met = task_metric(res[0], tokenizer, data_test)
    print(met)


if __name__ == '__main__':
    main()
    just_task()
