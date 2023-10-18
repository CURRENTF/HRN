# Multi-Defendant Legal Judgment Prediction via Hierarchical Reasoning
## Dataset
Datasets are located in `HRN/dataset/dataset-v5`. All samples have been anonymized.
- 18968 samples in train
- 2379 samples in valid
- 2370 samples in test

### Sample Structure
- fact: str # Fact description of the case
- interpretation: str # Judgement and conclusion
- laws: List[int] # Laws presented in "interpretation"
- relations: List[List[str]] # Relations between criminals
- criminals_info: Dict[str, Dict] # Relevant sentencing information of the defendant
  - name: str # Anonymous code of the defendant
    - accusations: List[str] # Crimes involved by the defendant
    - laws: List[int] # Laws involved by the defendant
    - states: List[int] # Situations involved by the defendant
    - term: str # The sentence term for the defendant

#### Example:
> {"fact": "......<LOC>区人民检察院指控：2017年1月起，被告人[被告A]在经营<LOC>区<LOC>252号酷乐台球室期间，在房间内摆放动物赌博游戏机1台（共8个机位，可同时供8人独立操作使用）、捕鱼赌博游戏机1台（共8个机位，可同时供8人独立操作使用）、斗地主赌博游戏机1台，并雇佣被告人[被告B]、[被告C]，以微信扫码充值、现金退换分的方式提供赌博活动。其中，被告人[被告B]主要负责维持赌场秩序、提供二维码供赌客扫赌资、为他人使用游戏机赌博提供服务；.......", 
> "interpretation": "本院认为：被告人[被告A]以营利为目的，......", 
> "laws": [303], 
> "relations": [["开设赌场罪", "B", "帮助", "A"], ["开设赌场罪", "C", "帮助", "A"]], 
> "criminals_info": 
>> {"[被告A]": {"accusations": ["开设赌场罪"], "laws": [303], "states": [6], "term": "有期徒刑六个月"}, "[被告B]": {"accusations": ["开设赌场罪"], "laws": [303], "states": [3, 6], "term": "拘役四个月，"}, "[被告C]": {"accusations": ["开设赌场罪"], "laws": [303], "states": [3, 5], "term": "拘役三个月，"}}}

## Reproduce Results

### Environment Setup

```bash
conda env create -f environment.yml
```

### Model Training

In `HRN/models/HRN` folder, run command below. Modify related args in `run_raw_data.sh` according to your needs.

```bash
bash config/run_raw_data.sh
```

### Model Testing

Modify related args in `test_raw_data.sh` according to your needs.
```bash
bash config/test_raw_data.sh
```