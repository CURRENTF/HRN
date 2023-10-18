import json
import pickle
import re
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
from transformers.trainer_utils import get_last_checkpoint
from src.model import FiDMT5
from src.text_match import re_match, tasks, ALL_LAWS, ALL_ACCUS, add_tokenizer_tokens, INV_STATES
from src.data_templates import apply_template
from sklearn import metrics
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


class ForTestTrainer(Seq2SeqTrainer):
    stage3_scores = []
    stage3r_scores = []
    stage_now = 0

    def set_stage(self, stage):
        self.stage_now = stage

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get("max_length") is not None else self.model.config.max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # ++++ my code +++++
        gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True
        # ---- end code -----

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        outputs = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # +++ my code +++
        generated_tokens = outputs.sequences
        probs = torch.stack(outputs.scores, dim=1).softmax(-1)
        # print('bos', self.tokenizer.bos_token_id)
        # print('eos', self.tokenizer.eos_token_id)
        # print('out', self.tokenizer.batch_decode(generated_tokens))
        # print(generated_tokens)
        gen_probs = torch.gather(probs, 2, generated_tokens[:, 1:, None]).squeeze(-1)
        sum_log_probs = torch.sum(torch.log(gen_probs), dim=-1, keepdim=False)
        if self.stage_now == 3:
            # print('for debug')
            # print(generated_tokens.shape)
            # print(sum_log_probs.shape)
            self.stage3_scores.extend(list(sum_log_probs.cpu().numpy()))
            # print(scores.shape)
            # for score in outputs.scores:
            #     # print("for debug", type(score))
            #     # print("for debug", generated_tokens.shape)
            #     # print("for debug", score[:, self.tokenizer.eos_token_id])
            #     # print("for debug", score)
            #     # self.stage3_scores.append(score[:, self.tokenizer.eos_token_id].cpu().numpy())
            #     self.stage3_scores.extend(list(score[:, self.tokenizer.eos_token_id].cpu().numpy()))
        elif self.stage_now == 4:
            self.stage3r_scores.extend(list(sum_log_probs.cpu().numpy()))
            # for score in outputs.scores:
            #     # self.stage3r_scores.append()
            #     self.stage3r_scores.extend(list(score[:, self.tokenizer.eos_token_id].cpu().numpy()))
        # --- my code ---

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)


def get_metrics(p, t):
    return {
        'acc': metrics.accuracy_score(t, p),
        'p': metrics.precision_score(t, p, average='macro'),
        'r': metrics.recall_score(t, p, average='macro'),
        'f1': metrics.f1_score(t, p, average='macro')
    }


correct, all_q = 0, 0


def main():
    # only raw data
    parser = HfArgumentParser((MyTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        my_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        my_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

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

    data_fields = {'test': f'../../dataset/dataset-v5/data_test_v5.01.jsonl'}
    raw_data = {}
    for k, v in data_fields.items():
        raw_data[k] = []
        with open(v, 'r', encoding='utf-8') as I:
            for line in I:
                raw_data[k].append(json.loads(line))

    def get_raw_states(sample):
        for name in sample['criminals_info']:
            sample['criminals_info'][name]['raw_states'] = []
            for state in sample['criminals_info'][name]['states']:
                sample['criminals_info'][name]['raw_states'].append(INV_STATES[state])
        return sample

    for k in tqdm(raw_data, desc='get raw states'):
        _ = []
        for d in raw_data[k]:
            _.append(get_raw_states(d))
        raw_data[k] = _

    def generate_diverse_samples(samples, test=False):
        res_list = []
        for sample in samples:
            res_list.extend(apply_template(sample, test=test))
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

    debug_merge_question = open('./debug/merge_question.txt', 'w', encoding='utf-8')
    eos_id = tokenizer.eos_token_id

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
        return {'input_ids': ids, 'attention_mask': attn}

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

    trainer = ForTestTrainer(
        model,
        training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=task_metric,
    )

    state_dict = torch.load(f'{training_args.resume_from_checkpoint}/pytorch_model.bin', map_location='cpu')
    trainer.model.load_state_dict(state_dict)

    relations = []
    states = []
    stage3_preds_labels = None
    stage3r_preds_labels = None

    for stage in range(4):

        from src.data_templates import strict_stages

        def convert_question(sample):
            global correct, all_q
            if stage > 1:
                n_question = strict_stages[stage]['in'].format(name=sample['name'],
                                                               relations=relations[sample['idx']],
                                                               states=states[sample['idx']])

            else:
                n_question = strict_stages[stage]['in'].format(name=sample['name'],
                                                               relations=relations[sample['idx']],
                                                               states='')
            if n_question == sample['question']:
                correct += 1
            all_q += 1

            return {'question': n_question}

        sub_data = DatasetDict()
        sub_data['test'] = Dataset.from_list(generate_diverse_samples(raw_data['test'], test=stage))
        data = sub_data
        if stage > 0:
            data = data.map(convert_question, batched=False, num_proc=my_args.num_proc)
        data = data.map(tokenize_question, num_proc=my_args.num_proc)
        data = data.map(tokenize_input, batched=True, batch_size=100, num_proc=my_args.num_proc)
        data = data.map(tokenize_output, batched=True, batch_size=100, num_proc=my_args.num_proc)
        data = data.map(merge_question, batched=False, num_proc=my_args.num_proc)
        # data_test = data['test'].to_list()
        data = data.remove_columns(['inputs', 'target', 'question', 'name'])
        trainer.set_stage(stage + 1)
        res = trainer.predict(data['test'], metric_key_prefix=f'stage{stage}')
        debug_output = open(f'./debug/stage{stage}.txt', 'w', encoding='utf-8')

        preds, labels, _ = res
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if stage == 2:
            stage3_preds_labels = res
        elif stage == 3:
            stage3r_preds_labels = res
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        for p, t in zip(preds, labels):
            print(p, file=debug_output)
            print(t, file=debug_output)
            print('-' * 80, file=debug_output)
        debug_output.close()

        if stage < 2:
            if stage == 0:
                for pred in preds:
                    res = re.search(r'他们之间的关系是(.*)', pred)
                    relations.append(res.group(1) if res else '')
                if len(relations) != len(data['test']):
                    raise ValueError("len(relations) != len(data['test'])")
            elif stage == 1:
                for pred in preds:
                    res = re.search(r'此被告存在的量刑情节为(.*)', pred)
                    states.append(res.group(1) if res else '')
                if len(states) != len(data['test']):
                    raise ValueError("len(states) != len(data['test'])")
        else:
            print(task_metric(res[:2]))
            print('-' * 80)

    dd = {
        'stage3_scores': trainer.stage3_scores,
        'stage3r_scores': trainer.stage3r_scores,
        'stage3_preds': stage3_preds_labels,
        'stage3r_preds': stage3r_preds_labels
    }
    pickle.dump(dd, open(f'./tmp/scores_and_preds.pkl', 'wb'))
    # print(correct, '/', all_q)
    debug_merge_question.close()
    max_score_test(task_metric)


def max_score_test(task_metric):
    dd = pickle.load(open('./tmp/scores_and_preds.pkl', 'rb'))
    print(type(dd['stage3_scores'][0]))
    std_preds = dd['stage3_preds']
    r_preds = dd['stage3r_preds']
    idx = 0
    for score, score_r in zip(dd['stage3_scores'], dd['stage3r_scores']):
        print(score, score_r)
        if score < score_r:
            std_preds.predictions[idx] = r_preds.predictions[idx]
        idx += 1
    print('max_score result:')
    print(task_metric([std_preds.predictions, std_preds.label_ids]))


if __name__ == '__main__':
    main()
