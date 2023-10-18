import json
import re

from peft import PeftModel
from tqdm import *
import requests.exceptions
from datasets import load_dataset, DatasetDict, Dataset
import sys
import os
from dataclasses import dataclass, field
from transformers import (HfArgumentParser, TrainingArguments, set_seed, AutoTokenizer, Trainer, MT5Tokenizer,
                          EvalPrediction, MT5Config, MT5ForConditionalGeneration, DataCollatorWithPadding,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_peft_available

from src.model import FiDMT5
from src.text_match import re_match, tasks, ALL_LAWS, ALL_ACCUS, add_tokenizer_tokens, INV_STATES
from src.data_templates import apply_template
from sklearn import metrics
import wandb
import numpy as np
from torch import nn
from typing import Dict, Union, Any, Optional, List, Tuple
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from src.cal_sample_weight import cal_weight
import src


@dataclass
class MyTrainingArguments:
    data_version: str = field(
        default='complete_v3'
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
        default=-1
    )
    load_pkl: bool = field(
        default=False
    )
    freeze_front_layers: int = field(
        default=0
    )
    data_mode: str = field(
        default='sub_data'
    )
    num_proc: int = field(
        default=4
    )
    test_batch_size: int = field(
        default=16
    )
    stage_weight: str = field(
        default=','.join([str(1.0)] * 4)
    )
    exclude_stage: Optional[Union[int, str]] = field(
        default=None
    )


class LossWeightTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def get_metrics(p, t):
    return {
        'acc': metrics.accuracy_score(t, p),
        'p': metrics.precision_score(t, p, average='macro'),
        'r': metrics.recall_score(t, p, average='macro'),
        'f1': metrics.f1_score(t, p, average='macro')
    }


def main():
    parser = HfArgumentParser((MyTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        my_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        my_args, training_args = parser.parse_args_into_dataclasses()

    my_args.stage_weight = [float(i) for i in my_args.stage_weight.split(',')]
    if my_args.exclude_stage is not None and isinstance(my_args.exclude_stage, str):
        if len(my_args.exclude_stage) == 1:
            try:
                my_args.exclude_stage = int(my_args.exclude_stage)
            except:
                pass

    set_seed(training_args.seed)
    if training_args.local_rank <= 0:
        wandb.init(
            project=f'HRN',
            name=f'model={my_args.model_name}_ctx={my_args.context_num}_ctx-len={my_args.per_context_len}_dataV={my_args.data_version}'
        )

    tokenizer = MT5Tokenizer.from_pretrained(my_args.model_name, use_fast=True)
    tokenizer = add_tokenizer_tokens(tokenizer)

    while True:
        try:
            model_cfg = MT5Config.from_pretrained(my_args.model_name)
            setattr(model_cfg, 'n_passages', my_args.context_num)
            setattr(model_cfg, 'passage_len', my_args.per_context_len)
            model = FiDMT5(model_cfg).cpu()
            raw_model = MT5ForConditionalGeneration.from_pretrained(my_args.model_name).cpu()
            model.load_t5(raw_model.state_dict())
            model = model.cuda()
            model.unwrap_encoder()
            model.resize_token_embeddings(250100 + 22 + 11 + 9 + 27 * 26)
            model.wrap_encoder()
            if 'mt5-base' in my_args.model_name and my_args.load_pkl:
                model.load_state_dict(torch.load('checkpoint/trained.pkl'))
            break
        except Exception as e:
            print(e)
            pass

    def split_fact(sample):
        ctxs = sample['ctxs']
        fact = "instruction: " + sample['question'] + 'context: '
        for d in ctxs:
            fact += d['text']
        return {
            'inputs': fact
        }

    def replace_criminal(sample):
        fact = sample['inputs']
        if '之间的关系是' not in sample['question']:
            c_name = re.search(r'\[被告[A-Z]{1,2}]', sample['question']).group()
            fact = fact.replace(c_name, '#' + c_name)
            return {'inputs': fact, 'target': sample['target'].replace('被告', '#' + c_name)}
        else:
            return {'inputs': fact, 'target': sample['target']}

    def tokenize_input(sample):
        res = tokenizer(
            sample['inputs'],
            max_length=my_args.context_num * my_args.per_context_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return res

    def tokenize_output(sample):
        return {
            'labels': tokenizer(
                sample['target'],
                max_length=64, padding=True, truncation=True, return_tensors='pt'
            )['input_ids'],
        }

    if my_args.data_mode == 'sub_data':
        data_fields = {}
        if training_args.do_train:
            data_fields['train'] = f'../../dataset/dataset-v5/sub_data/train_data_{my_args.data_version}.jsonl'
            data_fields['valid'] = f'../../dataset/dataset-v5/sub_data/valid_data_{my_args.data_version}.jsonl'
        if training_args.do_predict:
            data_fields['test'] = f'../../dataset/dataset-v5/sub_data/test_data_{my_args.data_version}.jsonl'
        data = load_dataset("json", data_files=data_fields)
        if my_args.valid_num > 0:
            data['valid'] = data['valid'].select(list(range(my_args.valid_num)))
        weights = cal_weight(data['train'], src.text_match.get_accus, 23)
        data = data.map(split_fact, batched=False, num_proc=my_args.num_proc)
        data = data.map(replace_criminal, batched=False, num_proc=my_args.num_proc)
    elif my_args.data_mode == 'raw_data':
        data_fields = {}
        if training_args.do_train:
            data_fields['train'] = f'../../dataset/dataset-v5/data_train_v5.01.jsonl'
            data_fields['valid'] = f'../../dataset/dataset-v5/data_valid_v5.01.jsonl'
        if training_args.do_predict:
            data_fields['test'] = f'../../dataset/dataset-v5/data_test_v5.01.jsonl'
        # data = load_dataset("json", data_files=data_fields)
        data = {}
        for k, v in data_fields.items():
            data[k] = []
            with open(v, 'r', encoding='utf-8') as I:
                for line in I:
                    data[k].append(json.loads(line))
        sub_data = DatasetDict()

        def get_raw_states(sample):
            for name in sample['criminals_info']:
                sample['criminals_info'][name]['raw_states'] = []
                for state in sample['criminals_info'][name]['states']:
                    sample['criminals_info'][name]['raw_states'].append(INV_STATES[state])
            return sample

        for k in tqdm(data, desc='get raw states'):
            _ = []
            for d in data[k]:
                _.append(get_raw_states(d))
            data[k] = _

        def generate_diverse_samples(samples):
            res_list = []
            for sample in samples:
                res_list.extend(apply_template(sample, exclude_stage=my_args.exclude_stage))
            return res_list

        def tokenize_question(sample):
            _ = tokenizer(
                sample['question'],
                padding='longest',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'question_ids': _['input_ids'],
                'question_attn': _['attention_mask']
            }

        if training_args.do_train:
            sub_data['train'] = Dataset.from_list(generate_diverse_samples(data['train']))
            sub_data['valid'] = Dataset.from_list(generate_diverse_samples(data['valid']))
        if training_args.do_predict:
            sub_data['test'] = Dataset.from_list(generate_diverse_samples(data['test']))
        data = sub_data
        data = data.map(tokenize_question, num_proc=my_args.num_proc)

        if my_args.use_weight:
            raise NotImplementedError

    else:
        raise NotImplementedError

    if training_args.do_train:
        with open('debug_preprocess.txt', 'w', encoding='utf-8') as O:
            print(data['train'][:2], file=O)
    data = data.map(tokenize_input, batched=True, batch_size=100, num_proc=my_args.num_proc)
    data = data.map(tokenize_output, batched=True, batch_size=100, num_proc=my_args.num_proc)
    data_test = None
    if my_args.data_mode == 'sub_data':
        data = data.remove_columns(['id', 'inputs', 'ctxs', 'target', 'answers', 'question'])
    elif my_args.data_mode == 'raw_data':
        eos_id = tokenizer.eos_token_id

        debug_merge_question = open('./debug/merge_question.txt', 'w', encoding='utf-8')

        def merge_question(sample):
            print(sample, file=debug_merge_question)
            q_ids, q_attn = torch.tensor(sample['question_ids'][0]), torch.tensor(sample['question_attn'][0])
            _ = torch.eq(q_ids, eos_id)
            _ = torch.nonzero(_)
            if _.shape[0] != 1:
                raise ValueError("eos != 1 ??")
            idx = _[0, 0]
            q_len = idx
            q_ids = torch.tensor(np.array([q_ids[:idx]] * my_args.context_num))
            q_attn = torch.tensor(np.array([q_attn[:idx]] * my_args.context_num))
            ids = torch.cat(
                [q_ids,
                 torch.tensor(sample['input_ids'])[:my_args.context_num * (my_args.per_context_len - q_len)]
                 .view(my_args.context_num, -1)],
                dim=1
            )[:, :my_args.per_context_len]
            attn = torch.cat(
                [q_attn,
                 torch.tensor(sample['attention_mask'])
                 [:my_args.context_num * (my_args.per_context_len - q_len)].view(my_args.context_num, -1)],
                dim=-1
            )[:, :my_args.per_context_len]

            return {'input_ids': ids, 'attention_mask': attn, 'loss_weight': my_args.stage_weight[sample['stage'] - 1]}

        data = data.map(merge_question, batched=False, num_proc=my_args.num_proc)
        debug_merge_question.close()
        data = data.remove_columns(['inputs', 'target', 'question', 'stage'])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # features = [data["train"][i] for i in range(2)]
    # print(data_collator(features))

    def task_metric(p) -> dict:
        preds, labels = p
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        p_res, t_res = {'laws': [], 'accus': [], 'term': []}, {'laws': [], 'accus': [], 'term': []}
        debug_cnt = 0
        debug_output = open(f'{training_args.output_dir}/debug_all.txt', 'w', encoding='utf-8')
        for p, t in zip(preds, labels):
            p_dict = re_match(p)
            t_dict = re_match(t)
            # debug_cnt += 1
            print(p, file=debug_output)
            print(t, file=debug_output)
            print(p_dict, file=debug_output)
            print(t_dict, file=debug_output)
            print('-' * 80, file=debug_output)
            # if debug_cnt==10:
            #     break
            for task in tasks:
                if task == 'term':
                    if t_dict[task] != -1:
                        if p_dict[task] == -1:
                            p_dict[task] = 8
                        p_res[task].append(p_dict[task])
                        t_res[task].append(t_dict[task])
                elif sum(t_dict[task]) > 0:
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

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=data['train'] if training_args.do_train else None,
        eval_dataset=data['valid'] if training_args.do_train else None,
        compute_metrics=task_metric,
    )

    for name, para in model.named_parameters():
        x = re.search(r'(encoder|decoder)\.block\.([0-9]+)', name)
        if x and int(x.group(2)) < my_args.freeze_front_layers:
            para.requires_grad = False

    if my_args.use_weight:
        def get_train_dataloader() -> DataLoader:
            self = trainer

            def seed_worker(_):
                worker_seed = torch.initial_seed() % 2 ** 32
                set_seed(worker_seed)

            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator

            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = WeightedRandomSampler(weights=weights,
                                                                     num_samples=my_args.weight_train_num,
                                                                     replacement=True)
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker

            dataloader = DataLoader(train_dataset, **dataloader_params)
            return self.accelerator.prepare(dataloader)

        trainer.get_train_dataloader = get_train_dataloader

    if training_args.do_train:
        trainer.train()

    if training_args.do_predict:
        res = trainer.predict(data['test'])
        print(res[-1])


if __name__ == '__main__':
    main()
