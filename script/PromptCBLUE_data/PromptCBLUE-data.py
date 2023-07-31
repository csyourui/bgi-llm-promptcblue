"""
Author: yourui
Date: 2023-06-19
Copyright (c) 2023 by yourui, Inc. All Rights Reserved.
"""
import argparse
import shutil
import hashlib
import json
import os
import random
import re
from typing import List, Tuple

import numpy as np


SEED = 1130


"""
usage: PromptCBLUE-data [-h] --cblue CBLUE --promptcblue PROMPTCBLUE --temp TEMP --out OUT

Build PromptCBLUE train data

optional arguments:
  -h, --help                    show this help message and exit
  --cblue CBLUE                 CBLUE data directory
  --promptcblue PROMPTCBLUE     promptcblue training data file
  --temp TEMP                   temp directory
  --out OUT                     output file directory
"""

"""
python PromptCBLUE-data.py \
    --cblue CBLUE \
    --promptcblue train.json \
    --temp data \
    --out data/BGI
"""
parser = argparse.ArgumentParser(prog='PromptCBLUE-data', description='Build PromptCBLUE train data')
parser.add_argument('--cblue', default="CBLUE", type=str, required=True, help="CBLUE data directory")
parser.add_argument('--promptcblue', default="train.json", type=str, required=True, help="promptcblue training data file")
parser.add_argument('--temp', default='data', type=str, required=True, help="temp directory")
parser.add_argument('--out',default='data/BGI', type=str, required=True, help="output file directory")

args = parser.parse_args()

CBLUE_DATA=args.cblue
PROMPTCBLUE_FILE=args.promptcblue
PROMPTCBLUE_DATA=args.temp
BGI_DATA=args.out

os.makedirs(PROMPTCBLUE_DATA, exist_ok=True)
os.makedirs(BGI_DATA, exist_ok=True)


# 原始训练集文件
FILE_ORIGIN_TRAIN_MAP = {
    "CMeIE": os.path.join(CBLUE_DATA, "CMeIE/CMeIE_train.json"),
    "CMeEE-V2": os.path.join(CBLUE_DATA, "CMeEE-V2/CMeEE-V2_train.json"),
    "KUAKE-QTR": os.path.join(CBLUE_DATA, "KUAKE-QTR/KUAKE-QTR_train.json"),
    "KUAKE-QIC": os.path.join(CBLUE_DATA, "KUAKE-QIC/KUAKE-QIC_train.json"),
    "CHIP-CDEE": os.path.join(CBLUE_DATA, "CHIP-CDEE/CHIP-CDEE_train.json"),
    "CHIP-CDN": os.path.join(CBLUE_DATA, "CHIP-CDN/CHIP-CDN_train.json"),
    "CHIP-CTC": os.path.join(CBLUE_DATA, "CHIP-CTC/CHIP-CTC_train.json"),
    "CHIP-STS": os.path.join(CBLUE_DATA, "CHIP-STS/CHIP-STS_train.json"),
    "IMCS-V2-NER": os.path.join(CBLUE_DATA, "IMCS-V2-NER/IMCS-V2_train.json"),
    "MedDG": os.path.join(CBLUE_DATA, "MedDG/MedDG_train.json"),
}
# 天池比赛数据分割后文件
FILE_PROMPT_ORIGIN_TRAIN_MAP = {
    "CMeIE": os.path.join(PROMPTCBLUE_DATA, "CMeIE/P_CMeIE.json"),
    "CMeEE-V2": os.path.join(PROMPTCBLUE_DATA, "CMeEE-V2/P_CMeEE-V2.json"),
    "KUAKE-IR": os.path.join(PROMPTCBLUE_DATA, "KUAKE-IR/P_KUAKE-IR.json"),
    "KUAKE-QTR": os.path.join(PROMPTCBLUE_DATA, "KUAKE-QTR/P_KUAKE-QTR.json"),
    "KUAKE-QIC": os.path.join(PROMPTCBLUE_DATA, "KUAKE-QIC/P_KUAKE-QIC.json"),
    "CHIP-CDEE": os.path.join(PROMPTCBLUE_DATA, "CHIP-CDEE/P_CHIP-CDEE.json"),
    "CHIP-CDN": os.path.join(PROMPTCBLUE_DATA, "CHIP-CDN/P_CHIP-CDN.json"),
    "CHIP-CTC": os.path.join(PROMPTCBLUE_DATA, "CHIP-CTC/P_CHIP-CTC.json"),
    "CHIP-STS": os.path.join(PROMPTCBLUE_DATA, "CHIP-STS/P_CHIP-STS.json"),
    "IMCS-V2-NER": os.path.join(PROMPTCBLUE_DATA, "IMCS-V2-NER/P_IMCS-V2-NER.json"),
    "MedDG": os.path.join(PROMPTCBLUE_DATA, "MedDG/P_MedDG.json"),
}
# 生成数据文件
FILE_BGI_PROMPT_TRAIN_MAP = {
    "CMeIE": os.path.join(BGI_DATA, "CMeIE_bgi_train_{NUM_SAMPLE}.json"),
    "CMeEE-V2": os.path.join(BGI_DATA, "CMeEE_bgi_train_{NUM_SAMPLE}.json"),
    "KUAKE-IR": os.path.join(BGI_DATA, "KUAKE-IR_bgi_train_{NUM_SAMPLE}.json"),
    "KUAKE-QTR": os.path.join(BGI_DATA, "KUAKE-QTR_bgi_train_{NUM_SAMPLE}.json"),
    "KUAKE-QIC": os.path.join(BGI_DATA, "KUAKE-QIC_bgi_train_{NUM_SAMPLE}.json"),
    "CHIP-CDEE": os.path.join(BGI_DATA, "CHIP-CDEE_bgi_train_{NUM_SAMPLE}.json"),
    "CHIP-CDN": os.path.join(BGI_DATA, "CHIP-CDN_bgi_train_{NUM_SAMPLE}.json"),
    "CHIP-CTC": os.path.join(BGI_DATA, "CHIP-CTC_bgi_train_{NUM_SAMPLE}.json"),
    "CHIP-STS": os.path.join(BGI_DATA, "CHIP-STS_bgi_train_{NUM_SAMPLE}.json"),
    "IMCS-V2-NER": os.path.join(BGI_DATA, "IMCS-V2-NER_bgi_train_{NUM_SAMPLE}.json"),
    "MedDG": os.path.join(BGI_DATA, "MedDG_bgi_train_{NUM_SAMPLE}.json"),
}

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def cal_file_hash(directory, output_file):
    def get_file_hash(file_path):
        h = hashlib.sha256()
        with open(file_path, 'rb') as file:
            while True:
                chunk = file.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def get_all_files(directory):
        for root, dirs, files in os.walk(directory):
            for file in sorted(files):
                if 'sha256' in file:
                    continue
                yield os.path.join(root, file)

    with open(os.path.join(directory, output_file), 'w') as f:
        for file_path in get_all_files(directory):
            file_path = file_path.replace('\\', '/')
            file_hash = get_file_hash(file_path)
            f.write(f'{file_path}\t{file_hash}\n')

def sample():
    sample_dist = {}
    with open("dev.json", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            input = obj["input"]
            obj["target"]
            type = obj["task_dataset"]
            if type not in sample_dist or (len(input) < sample_dist[type]["length"] and len(input) > 50):
                obj["length"] = len(input)
                sample_dist[type] = obj
    with open("sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_dist, f, ensure_ascii=False, indent=4)
    with open("md.md",  "w", encoding="utf-8") as w:
        w.write("| 任务名 | 输入 | 目标 | 选择项 | 任务类型 | 任务数据集  | 示例id |\n")
        w.write("| ----- | ----- | ----- | ----- | ----- | -----  | ----- |\n")
        for item in sample_dist:
            obj = sample_dist[item]
            line = f"| {item} | {obj['input']} | {obj['target']} | {obj['answer_choices']} | {obj['task_type']} | {obj['task_dataset']} | {obj['sample_id']} |"
            line = line.replace("\n", "\\n")
            w.write(line)
            w.write("\n")

def check_new_pattern(file, PATTERN_KEY):
    res = {}
    sum = 0

    with open(file, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            exist = False
            for key in PATTERN_KEY:
                if key not in res:
                    res[key] = 0
                if re.search(key, item['input']) is not None:
                    exist = True
                    res[key] = res[key] + 1
            if not exist:
                raise Exception(f'Not known pattern: {item}')
    for key in res:
        sum = sum + res[key]
    print(json.dumps(res, ensure_ascii=False, indent=4))
    if sum != len(obj):
        raise Exception(f'Not same size pattern: {sum} != {len(obj)}')
    print(f"total_sum:{sum}")

def gen_diff_item(mini_list: List[str], total_set: List[str], n = 1)-> List[str]:
    difference = sorted(set(total_set) - set(mini_list))
    if difference:
        chosen_elements = random.sample((difference), k=n)
        return chosen_elements
    else:
        return None

def random_insert(list1, list2):
    for item in list2:
        random_index = random.randint(0, len(list1))
        list1.insert(random_index, item)
    return list1

def split():
    sample_dist = {}
    with open(PROMPTCBLUE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["task_dataset"] not in sample_dist:
                list = []
                list.append(obj)
                sample_dist[obj["task_dataset"]] = list
            else:
                sample_dist[obj["task_dataset"]].append(obj)
    for item in sample_dist:
        print(item, len(sample_dist[item]))
        os.makedirs(os.path.join(PROMPTCBLUE_DATA, item), exist_ok=True)

        filename = os.path.join(PROMPTCBLUE_DATA, item, f"P_{item}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(sample_dist[item], f, ensure_ascii=False, indent=4)

def CMeIE(num_sample = 14291):
    set_random_seed(SEED)

    # 1、60.5%的概率加上'\n答：'
    # 2、25%的概率生成，'没有指定类型的三元组'问题，即设置一个或多个不存在的实体，则回答中全为空
    # 3、75%的问题中，要有47%的数据，自动补充一个多的不存在答案实体，回答中也只回答空

    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["CMeIE"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["CMeIE"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["CMeIE"].replace("{NUM_SAMPLE}", str(num_sample))
    P_EXIST = 0.75
    P_ADD_ANS=0.605
    P_NO_ANS=0.47

    def pattern_1(text, predicates):
        # 403
        return f"{text}\n这个句子里面具有一定医学关系的实体组有哪些？\n三元组关系选项：{predicates}"

    def pattern_2(text, predicates):
        # 403
        return f"找出句子中的具有{predicates}关系类型的头尾实体对：\n{text}"

    def pattern_3(text, predicates):
        # 454
        return f"{text}\n问题：句子中的{predicates}等关系类型三元组是什么？"

    def pattern_4(text, predicates):
        # 463
        return f"给出句子中的{predicates}等关系类型的实体对：{text}"

    def pattern_5(text, predicates):
        # 420
        return f"找出指定的三元组：\n{text}\n实体间关系：{predicates}"

    def pattern_6(text, predicates):
        # 414
        return f"同时完成实体识别与关系识别：\n{text}\n三元组关系类型：{predicates}"

    def pattern_7(text, predicates):
        # 443
        return f"根据给定的实体间的关系，抽取具有这些关系的实体对：\n{text}\n实体间关系标签：{predicates}"

    PATTERN_KEY = {
        '(.+)\n这个句子里面具有一定医学关系的实体组有哪些？\n三元组关系选项：(.+)':  pattern_1,
        '找出句子中的具有(.+)关系类型的头尾实体对：\n(.+)': pattern_2,
        '(.+)\n问题：句子中的(.+)等关系类型三元组是什么？': pattern_3,
        '给出句子中的(.+)等关系类型的实体对：(.+)': pattern_4,
        '找出指定的三元组：\n(.+)\n实体间关系：(.+)': pattern_5,
        '同时完成实体识别与关系识别：\n(.+)\n三元组关系类型：(.+)': pattern_6,
        '根据给定的实体间的关系，抽取具有这些关系的实体对：\n(.+)\n实体间关系标签：(.+)':pattern_7
    }

    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    predicate_set = set()
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for spo in item["spo_list"]:
                predicate_set.add(spo['predicate'])
    predicate_set = sorted(predicate_set)
    print(f"len(predicate_set) = {len(predicate_set)}")

    def pattern_handler(item):
        text = item["text"]
        answer_choices = [spo["predicate"] for spo in item["spo_list"]]
        answer_choices = sorted(set(answer_choices),key=answer_choices.index)
        target_text_list = []

        if random.random() < P_EXIST:
            #75%
            if random.random() < P_NO_ANS:
                chosen_element = gen_diff_item(answer_choices, predicate_set)
                if chosen_element is not None:
                    answer_choices.extend(chosen_element)

            for answer_choice in answer_choices:
                ans_str = f"具有{answer_choice}关系的头尾实体对如下："
                for spo in item["spo_list"]:
                    if spo['predicate'] == answer_choice:
                        ans_str = ans_str + f"头实体为{spo['subject']}，尾实体为{spo['object']['@value']}。"
                target_text_list.append(ans_str)
            target_text = "\n".join(target_text_list)
        else :
            #25%
            n = random.randint(1, 3)
            answer_choices = gen_diff_item(answer_choices, predicate_set, n)
            target_text = "没有指定类型的三元组"

        predicates = "，".join(answer_choices)
        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](text, predicates)
        if random.random() < P_ADD_ANS:
            input_text = input_text + "\n答："

        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": answer_choices,
            "task_type": "spo_generation",
            "task_dataset": "CMeIE",
            "sample_id": "bgi"
        }
        return processed_data


    train_sample = []
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            if '﻿' in line:
                continue
            item = json.loads(line)
            train_sample.append(pattern_handler(item))
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def CMeEE_V2(num_sample = 15000):
    set_random_seed(SEED)

    # 1、60%的概率加上'\n答：'
    # 2、7%的概率生成，'上述句子没有指定类型实体'问题，即设置一个或多个不存在的实体，则回答中全为空
    # 3、93%的问题中，要有42%的数据，自动补充一个多的不存在答案实体，回答中也只回答空
    P_EXIST = 0.93
    P_ADD_ANS=0.60
    P_NO_ANS=0.42
    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["CMeEE-V2"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["CMeEE-V2"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["CMeEE-V2"].replace("{NUM_SAMPLE}", str(num_sample))

    def pattern_1(text, predicates):
        # 430
        return f"找出指定的实体：\n{text}\n类型选项：{predicates}"
    def pattern_2(text, predicates):
        # 424
        return f"找出指定的实体：\n{text}\n实体类型选项：{predicates}"
    def pattern_3(text, predicates):
        # 471
        return f"给出句子中的实体：\n{text}\n医学实体选项：{predicates}"
    def pattern_4(text, predicates):
        # 458
        return f"实体识别：\n{text}\n实体类型：{predicates}"
    def pattern_5(text, predicates):
        # 443
        return f"{text}\n这个句子里面实体有哪些？\n实体选项：{predicates}"
    def pattern_6(text, predicates):
        # 486
        return f"实体抽取：\n{text}\n选项：{predicates}"
    def pattern_7(text, predicates):
        # 445
        return f"医学实体识别：\n{text}\n实体选项：{predicates}"
    def pattern_8(text, predicates):
        # 474
        return f"{text}\n问题：句子中的{predicates}实体是什么？"
    def pattern_9(text, predicates):
        # 479
        return f"下面句子中的{predicates}实体有哪些？\n{text}"
    def pattern_10(text, predicates):
        # 414
        return f"找出句子中的{predicates}实体：\n{text}"
    def pattern_11(text, predicates):
        # 476
        return f"生成句子中的{predicates}实体：\n{text}"

    PATTERN_KEY = {
        '找出指定的实体：\n(.+)\n类型选项：(.+)':  pattern_1,
        '找出指定的实体：\n(.+)\n实体类型选项：(.+)':  pattern_2,
        '给出句子中的实体：\n(.+)\n医学实体选项：(.+)':  pattern_3,
        '实体识别：\n(.+)\n实体类型：(.+)':  pattern_4,
        '(.+)\n这个句子里面实体有哪些？\n实体选项：(.+)':  pattern_5,
        '实体抽取：\n(.+)\n选项：(.+)':  pattern_6,
        '医学实体识别：\n(.+)\n实体选项：(.+)':  pattern_7,
        '(.+)\n问题：句子中的(.+)实体是什么？':  pattern_8,
        '下面句子中的(.+)实体有哪些？\n(.+)':  pattern_9,
        '找出句子中的(.+)实体：\n(.+)':  pattern_10,
        '生成句子中的(.+)实体：\n(.+)':  pattern_11,
    }
    ENTITY_MAP={
        "dis":"疾病",
        "sym":"临床表现",
        "pro":"医疗程序",
        "equ":"医疗设备",
        "dru":"药物",
        "ite":"医学检验项目",
        "bod":"身体部位",
        "dep":"医院科室",
        "mic":"微生物类"
    }
    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)
    entity_set = set()
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            for entity in item["entities"]:
                entity_set.add(ENTITY_MAP[entity['type']])
    entity_set = sorted(entity_set)
    print(f"entity_set:{entity_set}")
    print(f"len(entity_set) = {len(entity_set)}")


    def pattern_handler(item):
        text = item["text"]
        entities_choices = [ENTITY_MAP[entity['type']] for entity in item["entities"]]
        entities_choices = sorted(set(entities_choices),key=entities_choices.index)
        target_text_list = []

        if random.random() < P_EXIST:
            if random.random() < P_NO_ANS:
                chosen_element = gen_diff_item(entities_choices, entity_set)
                if chosen_element is not None:
                    if len(entities_choices) < 1:
                        entities_choices.insert(0, chosen_element[0])
                    else:
                        entities_choices.insert(random.randint(0, len(entities_choices) - 1), chosen_element[0])

            for entities_choice in entities_choices:
                entities_value_set = set()
                for entity in item["entities"]:
                    if entities_choice == ENTITY_MAP[entity["type"]]:
                        entities_value_set.add(entity["entity"])
                entities_str = f"{entities_choice}实体：" + "，".join(sorted(entities_value_set))
                target_text_list.append(entities_str)
            target_text = "\n".join(target_text_list)
            target_text = "上述句子中的实体包含：\n" + target_text
        else :
            entities_choices = gen_diff_item(entities_choices, entity_set, 1)
            target_text = "上述句子没有指定类型实体"

        entities_chr = "，".join(entities_choices)
        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](text, entities_chr)
        if random.random() < P_ADD_ANS:
            input_text = input_text + "\n答："

        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": entities_choices,
            "task_type": "ner",
            "task_dataset": "CMeEE-V2",
            "sample_id": "bgi"
        }
        return processed_data


    train_sample = []
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            train_sample.append(pattern_handler(item))
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def KUAKE_IR(num_sample = 10000):
    set_random_seed(SEED)
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["KUAKE-IR"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["KUAKE-IR"].replace("{NUM_SAMPLE}", str(num_sample))
    P_ADD_ANS = 0.6
    P_RELEVENT = 0.34
    def pattern_1(query, doc):
        return f"医疗搜索：{query}\n以下回答内容是否能够回答搜索问题？\n回答内容：{doc}\n选项: 相关，不相关"
    def pattern_2(query, doc):
        return f"医疗搜索：{query}\n回答内容：{doc}\n上述搜索和回答是否相关？\n选项: 相关，不相关"
    def pattern_3(query, doc):
        return f"以下回答内容是否与这里的医疗搜索相关？\n医疗搜索：{query}\n回答内容：{doc}\n选项: 相关，不相关"
    PATTERN_KEY = {
        '医疗搜索：(.+)\n以下回答内容是否能够回答搜索问题？\n回答内容：(.+)\n选项: 相关，不相关': pattern_1,
        '医疗搜索：(.+)\n回答内容：(.+)\n上述搜索和回答是否相关？\n选项: 相关，不相关': pattern_2,
        '以下回答内容是否与这里的医疗搜索相关？\n医疗搜索：(.+)\n回答内容：(.+)\n选项: 相关，不相关': pattern_3,
    }

    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    doc_id2doc = {}
    query_id2query = {}
    query_id2doc_id = {}
    with open(os.path.join(CBLUE_DATA, "KUAKE-IR/corpus.tsv"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split('\t')
            doc_id2doc[int(parts[0])] = parts[1]
    with open(os.path.join(CBLUE_DATA, "KUAKE-IR/KUAKE-IR_train_query.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split('\t')
            query_id2query[int(parts[0])] = parts[1]
    with open(os.path.join(CBLUE_DATA, "KUAKE-IR/KUAKE-IR_train.tsv"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split('\t')
            query_id2doc_id[int(parts[0])] = int(parts[1])

    doc_ids = list(doc_id2doc.keys())
    def generate_pairs(query_id):
        def random_except(exception):
            while True:
                random_id = random.choice(doc_ids)
                if random_id != exception:
                    return random_id

        query = query_id2query[query_id]
        doc_id = query_id2doc_id[query_id]
        neg_doc_id = random_except(doc_id)
        query_docs = {
            "query_id": query_id,
            "query":query,
            "doc_id":doc_id,
            "doc":doc_id2doc[doc_id],
            "neg_doc_id":neg_doc_id,
            "neg_doc":doc_id2doc[neg_doc_id]
        }
        return query_docs

    query_docs = [generate_pairs(query_id) for query_id in range(1, len(query_id2query) + 1)]

    def pattern_handler(item):
        query = item['query']
        doc = ""
        if random.random() < P_RELEVENT:
            target_text = "相关"
            doc = item['doc']
        else:
            target_text = "不相关"
            doc = item['neg_doc']
        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](query, doc)
        if random.random() < P_ADD_ANS:
            input_text = input_text + "\n答："
        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": ["相关","不相关"],
            "task_type": "ner",
            "task_dataset": "CMeEE-V2",
            "sample_id": "bgi"
        }
        return processed_data

    train_sample = []
    for item in query_docs:
        train_sample.append(pattern_handler(item))
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def KUAKE_QTR(num_sample = 24174):
    set_random_seed(SEED)
    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["KUAKE-QTR"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["KUAKE-QTR"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["KUAKE-QTR"].replace("{NUM_SAMPLE}", str(num_sample))
    P_ADD_ANS = 0.6
    def pattern_1(query, title, choices):
        return f"下面的搜索词和页面标签的意思有多相同？\n搜索词：{query}\n页面标签：{title}\n选项：{choices}"
    def pattern_2(query, title, choices):
        return f"下面两个句子的语义相似程度是{choices}中的哪一种？\n“{query}”,“{title}”"
    def pattern_3(query, title, choices):
        return f"“{query}”和“{title}”的意思有多相似？\n选项：{choices}"
    def pattern_4(query, title, choices):
        return f"以下两句话的意思相同的吗？\n“{query}”，“{title}”。\n选项：{choices}"
    def pattern_5(query, title, choices):
        return f"搜索词：“{query}”。页面标题：“{title}”。这两句是一样的意思吗？选项：{choices}"
    def pattern_6(query, title, choices):
        return f"“{query}”和“{title}”是同一个意思吗？\n选项：{choices}"
    def pattern_7(query, title, choices):
        return f"“{query}”，“{title}”。\n这两句话的意思的匹配程度如何？\n选项：{choices}"
    def pattern_8(query, title, choices):
        return f"我想知道下面两句话的意思有多相似。\n“{query}”，“{title}”\n选项：{choices}"
    PATTERN_KEY = {
        '下面的搜索词和页面标签的意思有多相同？\n搜索词：(.+)\n页面标签：(.+)\n选项：(.+)':  pattern_1,
        '下面两个句子的语义相似程度是(.+)中的哪一种？\n“(.+)”，“(.+)”':  pattern_2,
        '“(.+)”和“(.+)”的意思有多相似？\n选项：(.+)':  pattern_3,
        '以下两句话的意思相同的吗？\n“(.+)”，“(.+)”。\n选项：(.+)': pattern_4,
        '搜索词：“(.+)”。页面标题：“(.+)”。这两句是一样的意思吗？选项：(.+)': pattern_5,
        '“(.+)”和“(.+)”是同一个意思吗？\n选项：(.+)':  pattern_6,
        '“(.+)”，“(.+)”。\n这两句话的意思的匹配程度如何？\n选项：(.+)':  pattern_7,
        '我想知道下面两句话的意思有多相似。\n“(.+)”，“(.+)”\n选项：(.+)': pattern_8,
    }
    answer_choices = ["完全不匹配或者没有参考价值", "很少匹配有一些参考价值", "部分匹配", "完全匹配"]

    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    def pattern_handler(item):
        query = item["query"]
        title = item["title"]
        target_text = answer_choices[int(item["label"])]
        choices = "，".join(answer_choices)

        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](query, title, choices)
        if random.random() < P_ADD_ANS:
            input_text = input_text + "\n答："

        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": answer_choices,
            "task_type": "matching",
            "task_dataset": "KUAKE-QTR",
            "sample_id": "bgi"
        }
        return processed_data

    train_sample = []
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            train_sample.append(pattern_handler(item))
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def KUAKE_QIC(num_sample = 5314):
    set_random_seed(SEED)

    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["KUAKE-QIC"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["KUAKE-QIC"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["KUAKE-QIC"].replace("{NUM_SAMPLE}", str(num_sample))

    P_ADD_ANS = 0.6
    P_ADD_NOT_EXIST = 0.1
    def pattern_1(query, choices):
        return f"{query}\n这个搜索是什么意图？\n类型选项：{choices}"
    def pattern_2(query, choices):
        return f"请问是什么意图类型？\n{query}\n搜索意图选项：{choices}"
    def pattern_3(query, choices):
        return f"{query}\n这个医疗搜索词是什么意图分类？\n选项：{choices}"
    def pattern_4(query, choices):
        return f"确定检索词的类型：\n{query}\n类型选项：{choices}"
    def pattern_5(query, choices):
        return f"判断下面搜索词的意图：\n{query}\n选项：{choices}"
    PATTERN_KEY = {
        '(.+)\n这个搜索是什么意图？\n类型选项：(.+)':  pattern_1,
        '请问是什么意图类型？\n(.+)\n搜索意图选项：(.+)':  pattern_2,
        '(.+)\n这个医疗搜索词是什么意图分类？\n选项：(.+)':  pattern_3,
        '确定检索词的类型：\n(.+)\n类型选项：(.+)': pattern_4,
        '判断下面搜索词的意图：\n(.+)\n选项：(.+)': pattern_5,
    }
    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    sample = {}
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            if item['label'] not in sample:
                sample[item['label']] = [item]
            else:
                sample[item['label']].append(item)
    del sample['其他']
    label_set = list(sample)
    print(f"label_set = {label_set}")
    print(f"len(label_set) = {len(label_set)}")

    def pattern_handler(item):
        query = item["query"]
        target_text = item["label"]
        if random.random() < P_ADD_NOT_EXIST:
            answer_choices = gen_diff_item([target_text], label_set, random.randint(3, 9))
            target_text = "非上述类型"
        else:
            answer_choices = gen_diff_item([target_text], label_set, random.randint(2, 9))
            answer_choices.append(target_text)

        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](query, "，".join(answer_choices))
        if random.random() < P_ADD_ANS:
            input_text = input_text + "\n答："

        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": answer_choices,
            "task_type": "cls",
            "task_dataset": "KUAKE-QIC",
            "sample_id": "bgi"
        }
        return processed_data

    train_sample = []
    for key in sample:
        for item in sample[key]:
            train_sample.append(pattern_handler(item))
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=5000)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def KUAKE_QQR(num_sample = 5000):
    set_random_seed(SEED)

def CHIP_CDEE(num_sample = 1587):
    """
    {
        "core_name": "疼痛",      # 主体词
        "tendency": "",           # 发生状态
        "character": [],          # 描述词
        "anatomy_list": [         # 解剖部位
          "右下腹"
        ]
    }
    """

    set_random_seed(SEED)
    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["CHIP-CDEE"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["CHIP-CDEE"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["CHIP-CDEE"].replace("{NUM_SAMPLE}", str(num_sample))

    P_ADD_ANS = 0.6
    def pattern_1(text):
        return f"生成句子中的临床发现事件属性是：\n{text}\n说明：临床发现事件的主体词包含发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值"
    def pattern_2(text):
        return f"{text}\n这个句子里面临床发现事件是？\n说明：临床发现事件由主体词，发生状态，描述词和解剖部位组成"
    def pattern_3(text):
        return f"临床发现事件抽取：{text}\n说明：临床发现事件的主体词包含发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值"
    def pattern_4(text):
        return f"找出句子中的临床发现事件及其属性：\n{text}\n说明：临床发现事件的主体词包含发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值"
    def pattern_5(text):
        return f"找出指定的临床发现事件属性：\n{text}\n事件抽取说明：临床发现事件由主体词，发生状态，描述词和解剖部位组成"
    def pattern_6(text):
        return f"'临床发现事件抽取：\n{text}\n说明：临床发现事件的主体词包含发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值"
    def pattern_7(text):
        return f"{text}\n问题：句子中的临床发现事件及其属性是什么？\n说明：临床发现事件由主体词，发生状态，描述词和解剖部位组成"
    PATTERN_KEY = {
        '生成句子中的临床发现事件属性是：\n(.+)\n说明：临床发现事件的主体词包含发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值':  pattern_1,
        '(.+)\n这个句子里面临床发现事件是？\n说明：临床发现事件由主体词，发生状态，描述词和解剖部位组成':  pattern_2,
        '临床发现事件抽取：(.+)\n说明：临床发现事件的主体词包含发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值': pattern_3,
        '找出句子中的临床发现事件及其属性：\n(.+)\n说明：临床发现事件的主体词包含发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值': pattern_4,
        '找出指定的临床发现事件属性：\n(.+)\n事件抽取说明：临床发现事件由主体词，发生状态，描述词和解剖部位组成': pattern_5,
        '临床发现事件抽取：\n(.+)\n说明：临床发现事件的主体词包含发生状态，描述词和解剖部位这三种属性，其中描述词和解剖部位可能有多个值': pattern_6,
        '(.+)\n问题：句子中的临床发现事件及其属性是什么？\n说明：临床发现事件由主体词，发生状态，描述词和解剖部位组成': pattern_7
    }

    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    def pattern_handler(item):
        text = item["text"]
        target_text_list = []
        for event in item["event"]:
            core_name = event["core_name"]
            tendency = event["tendency"]
            character = "，".join(event["character"])
            anatomy_list = "，".join(event["anatomy_list"])
            target_text_list.append(f"主体词：{core_name}；发生状态：{tendency}；描述词：{character}；解剖部位：{anatomy_list}")
        target_text = "上述句子中的临床发现事件如下：\n" + "\n".join(target_text_list)

        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](text)
        if random.random() < P_ADD_ANS:
            input_text = input_text + "\n答："
        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": None,
            "task_type": "event_extraction",
            "task_dataset": "CHIP-CDEE",
            "sample_id": "bgi"
        }
        return processed_data
    train_sample = []
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            train_sample.append(pattern_handler(item))
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def CHIP_CDN(num_sample = 6000):
    set_random_seed(SEED)
    # 1、60%的概率加上'\n答：'
    # 2、15%的概率生成，'没有对应的标准化实体'问题，即设置一个或多个不存在的实体，则回答中全为空
    # 3、85%的问题中

    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["CHIP-CDN"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["CHIP-CDN"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["CHIP-CDN"].replace("{NUM_SAMPLE}", str(num_sample))

    P_ADD_ANS = 0.6
    P_EXIST = 0.85
    def pattern_1(text, choices):
        return f"诊断实体的语义标准化：\n{text}\n实体选项：{choices}\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词"
    def pattern_2(text, choices):
        return f"找出归一后的标准词：\n{text}\n选项：{choices}\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词"
    def pattern_3(text, choices):
        return f"诊断归一化：\n{text}\n选项：{choices}\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词"
    def pattern_4(text, choices):
        return f"给出下面诊断原词的标准化：\n{text}\n候选集：{choices}\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词"
    def pattern_5(text, choices):
        return f"给出诊断的归一化：\n{text}\n医学实体选项：{choices}\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词"
    def pattern_6(text, choices):
        return f"实体归一化：\n{text}\n实体候选：{choices}\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词"
    def pattern_7(text, choices):
        return f"{text}\n归一化后的标准词是？\n实体选项：{choices}\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词"
    PATTERN_KEY = {
        '诊断实体的语义标准化：\n(.+)\n实体选项：(.+)\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词':  pattern_1,
        '找出归一后的标准词：\n(.+)\n选项：(.+)\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词':  pattern_2,
        '诊断归一化：\n(.+)\n选项：(.+)\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词':  pattern_3,
        '给出下面诊断原词的标准化：\n(.+)\n候选集：(.+)\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词':  pattern_4,
        '给出诊断的归一化：\n(.+)\n医学实体选项：(.+)\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词':  pattern_5,
        '实体归一化：\n(.+)\n实体候选：(.+)\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词':  pattern_6,
        '(.+)\n归一化后的标准词是？\n实体选项：(.+)\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词':  pattern_7,
    }

    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    normalized_set = set()
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            normalized_result = item["normalized_result"].strip('\"').split("##")
            normalized_set.update(normalized_result)
    normalized_set = sorted(normalized_set)
    print(f"len(normalized_set) = {len(normalized_set)}")

    def pattern_handler(item):
        text = item["text"]
        normalized_result = item["normalized_result"].strip('\"').split("##")
        target_text = "，".join(normalized_result)

        if random.random() < P_EXIST:
            chosen_element = gen_diff_item(normalized_result, normalized_set, random.randint(1, 3))
            normalized_result = random_insert(normalized_result, chosen_element)

        else :
            target_text = "没有指定类型的三元组"
            normalized_result = gen_diff_item(normalized_result, normalized_set, random.randint(2, 10))

        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](text, "，".join(normalized_result))
        if random.random() < P_ADD_ANS:
            input_text = input_text + "\n答："

        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": normalized_result,
            "task_type": "normalization",
            "task_dataset": "CHIP-CDN",
            "sample_id": "bgi"
        }
        return processed_data


    train_sample = []
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            train_sample.append(pattern_handler(item))
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def CHIP_CTC(num_sample = 22962):
    set_random_seed(SEED)
    # 1、60%的概率加上'\n答：'
    # 2、4%的概率生成，'非上述类型'问题，即设置一个或多个不存在的实体，则回答中全为空
    # 3、96%的问题中

    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["CHIP-CTC"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["CHIP-CTC"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["CHIP-CTC"].replace("{NUM_SAMPLE}", str(num_sample))

    P_ADD_ANS = 0.6
    P_EXIST = 0.96
    def pattern_1(text, choices):
        return f"{text}\n这句话是什么临床试验筛选标准类型？\n类型选项：{choices}"
    def pattern_2(text, choices):
        return f"判断临床试验筛选标准的类型：\n{text}\n选项：{choices}"
    def pattern_3(text, choices):
        return f"请问是什么类型？\n{text}\n临床试验筛选标准选项：{choices}"
    def pattern_4(text, choices):
        return f"确定试验筛选标准的类型：\n{text}\n类型选项：{choices}"
    def pattern_5(text, choices):
        return f"{text}\n是什么临床试验筛选标准类型？\n选项：{choices}"
    PATTERN_KEY = {
        '(.+)\n这句话是什么临床试验筛选标准类型？\n类型选项：(.+)':  pattern_1,
        '判断临床试验筛选标准的类型：\n(.+)\n选项：(.+)': pattern_2,
        '请问是什么类型？\n(.+)\n临床试验筛选标准选项：(.+)': pattern_3,
        '确定试验筛选标准的类型：\n(.+)\n类型选项：(.+)': pattern_4,
        '(.+)\n是什么临床试验筛选标准类型？\n选项：(.+)': pattern_5
    }

    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    translation_map = {
        "Disease": "疾病",
        "Symptom": "症状(患者感受)",
        "Sign": "体征(医生检测）",
        "Pregnancy-related Activity": "怀孕相关",
        "Neoplasm Status": "肿瘤进展",
        "Non-Neoplasm Disease Stage": "疾病分期",
        "Allergy Intolerance": "过敏耐受",
        "Organ or Tissue Status": "器官组织状态",
        "Life Expectancy": "预期寿命",
        "Oral related": "口腔相关",
        "Pharmaceutical Substance or Drug": "药物",
        "Therapy or Surgery": "治疗或手术",
        "Device": "设备",
        "Nursing": "护理",
        "Diagnostic": "诊断",
        "Laboratory Examinations": "实验室检查",
        "Risk Assessment": "风险评估",
        "Receptor Status": "受体状态",
        "Age": "年龄",
        "Special Patient Characteristic": "特殊病人特征",
        "Literacy": "读写能力",
        "Gender": "性别",
        "Education": "教育情况",
        "Address": "居住情况",
        "Ethnicity": "种族",
        "Consent": "知情同意",
        "Enrollment in other studies": "参与其它试验",
        "Researcher Decision": "研究者决定",
        "Capacity": "能力",
        "Ethical Audit": "伦理审查",
        "Compliance with Protocol": "依存性",
        "Addictive Behavior": "成瘾行为",
        "Bedtime": "睡眠",
        "Exercise": "锻炼",
        "Diet": "饮食",
        "Alcohol Consumer": "酒精使用",
        "Sexual related": "性取向",
        "Smoking Status": "吸烟状况",
        "Blood Donation": "献血",
        "Encounter": "病例来源",
        "Disabilities": "残疾群体",
        "Healthy": "健康群体",
        "Data Accessible": "数据可及性",
        "Multiple": "含有多类别的语句",
    }

    label_list = list(translation_map.values())

    cal = {}
    with open(FILE_PROMPT_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            length = len(item['answer_choices'])
            if length not in cal:
                cal[length] = 1
            else:
                cal[length] = cal[length] + 1
    sorted_map = {k: cal[k] for k in sorted(cal)}
    total_sum = sum(sorted_map.values())
    probability_distribution = {key: value / total_sum for key, value in sorted_map.items()}
    values = np.array(list(sorted_map.keys()), dtype=int) - 1
    probabilities = np.array(list(probability_distribution.values()), dtype=float)

    def pattern_handler(item):
        text = item["text"].strip()
        label = translation_map[item["label"]]

        if random.random() < P_EXIST:
            target_text = label
            answer_choices = gen_diff_item([label], label_list, np.random.choice(values,p=probabilities))
            answer_choices = random_insert(answer_choices, [label])
        else :
            target_text = "非上述类型"
            answer_choices = gen_diff_item([label], label_list, np.random.choice(values,p=probabilities) + 1)

        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](text, "，".join(answer_choices))
        if random.random() < P_ADD_ANS:
            input_text = input_text + "\n答："

        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": answer_choices,
            "task_type": "normalization",
            "task_dataset": "CHIP-CDN",
            "sample_id": "bgi"
        }
        return processed_data


    train_sample = []
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            train_sample.append(pattern_handler(item))
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def CHIP_MDCFNPC(num_sample = 1000):
    set_random_seed(SEED)

def CHIP_STS(num_sample = 16000):
    set_random_seed(SEED)
    # 1、60%的概率加上'\n答：'
    # 2、4%的概率生成，'非上述类型'问题，即设置一个或多个不存在的实体，则回答中全为空
    # 3、96%的问题中

    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["CHIP-STS"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["CHIP-STS"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["CHIP-STS"].replace("{NUM_SAMPLE}", str(num_sample))

    def pattern_1(text1, text2):
        return f"“{text1}”，“{text2}”。\n这两句是一样的意思吗？\n选项：是的，不是\n答："
    def pattern_2(text1, text2):
        return f"我是否可以用以下的句子：“{text1}”，来替换这个句子：“{text2}”，并且它们有相同的意思？\n选项：是的，不是\n答："
    def pattern_3(text1, text2):
        return f"“{text1}”和“{text2}”是同一个意思吗？\n选项：是的，不是\n答："
    def pattern_4(text1, text2):
        return f"以下两句话的意思相同的吗？\n“{text1}”，“{text2}”\n选项：是的，不是\n答："
    def pattern_5(text1, text2):
        return f"下面两个句子语义是“相同”或“不同”？\n“{text1}”，“{text2}”。\n选项：相同，不同\n答："
    def pattern_6(text1, text2):
        return f"我想知道下面两句话的意思是否相同。\n“{text1}”，“{text2}”。\n选项：是的，不是\n答："
    PATTERN_KEY = {
        '“(.+)”，“(.+)”。\n这两句是一样的意思吗？\n选项：是的，不是\n答：':  pattern_1,
        '我是否可以用以下的句子：“(.+)”，来替换这个句子：“(.+)”，并且它们有相同的意思？\n选项：是的，不是\n答：': pattern_2,
        '“(.+)”和“(.+)”是同一个意思吗？\n选项：是的，不是\n答：': pattern_3,
        '以下两句话的意思相同的吗？\n“(.+)”，“(.+)”\n选项：是的，不是\n答：': pattern_4,
        '下面两个句子语义是“相同”或“不同”？\n“(.+)”，“(.+)”。\n选项：相同，不同\n答：': pattern_5,
        '我想知道下面两句话的意思是否相同。\n“(.+)”，“(.+)”。\n选项：是的，不是\n答：': pattern_6,
    }

    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    def pattern_handler(item):
        text1 = item["text1"]
        text2 = item["text2"]
        label = item["label"]

        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](text1, text2)

        if input_text.endswith("\n选项：相同，不同\n答："):
            target_text = "相同" if label == "1" else "不同"
            answer_choices = ["相同", "不同"]
        elif input_text.endswith("\n选项：是的，不是\n答："):
            target_text = "是的" if label == "1" else "不是"
            answer_choices = ["是的", "不是"]
        else:
            raise Exception(f'Error input_text: {input_text}')

        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": answer_choices,
            "task_type": "cls",
            "task_dataset": "CHIP-STS",
            "sample_id": "bgi"
        }
        return processed_data

    train_sample = []
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            train_sample.append(pattern_handler(item))
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def IMCS_V2_NER(num_sample = 41765):
    set_random_seed(SEED)

    # 1、60.5%的概率加上'\n答：'
    # 2、12%的概率生成，'上述句子没有指定类型实体'问题，即设置一个或多个不存在的实体，则回答中全为空
    # 3、88%的问题中，要有50%的数据，自动补充一个多的不存在答案实体，回答中也只回答空

    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["IMCS-V2-NER"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["IMCS-V2-NER"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["IMCS-V2-NER"].replace("{NUM_SAMPLE}", str(num_sample))

    P_ADD_ANS = 0.6
    P_EXIST = 0.88
    P_NO_ANS = 0.5
    def pattern_1(sentence, choices):
        return f"找出下面问诊语句中的{choices}实体：\n{sentence}"
    def pattern_2(sentence, choices):
        return f"找出下面问诊语句中的{choices}实体：\n{sentence}"
    def pattern_3(sentence, choices):
        return f"下面对话中的{choices}有哪些？\n{sentence}"
    def pattern_4(sentence, choices):
        return f"{sentence}\n问题：上述对话中的{choices}实体是哪些？"
    def pattern_5(sentence, choices):
        return f"问诊对话的实体抽取：{sentence}\n选项：{choices}"
    def pattern_6(sentence, choices):
        return f"{sentence}\n上述问诊中的{choices}实体是什么？"
    PATTERN_KEY = {
        '找出下面问诊语句中的(.+)实体：\n(.+)': pattern_1,
        '找出下面句子中的(.+)实体：\n(.+)': pattern_2,
        '下面对话中的(.+)有哪些？\n(.+)': pattern_3,
        '(.+)\n问题：上述对话中的(.+)实体是哪些？': pattern_4,
        '问诊对话的实体抽取：(.+)\n选项：(.+)': pattern_5,
        '(.+)\n上述问诊中的(.+)实体是什么？': pattern_6,
    }

    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    TYPE_TO_ENTITY = ["症状", "具体的药物名称", "药物类别", "医学检查检验", "医疗操作"]
    ENTITY_MAP = {
        "B-Symptom":"症状",
        "I-Symptom":"症状",
        "B-Drug":"具体的药物名称",
        "I-Drug":"具体的药物名称",
        "B-Drug_Category":"药物类别",
        "I-Drug_Category":"药物类别",
        "B-Medical_Examination":"医学检查检验",
        "I-Medical_Examination":"医学检查检验",
        "B-Operation":"医疗操作",
        "I-Operation":"医疗操作",
    }

    def get_symptoms_map(sentence:str, BIO_label:list[str]):
        symptoms = {}
        length = len(sentence)
        assert length == len(BIO_label)
        for key in TYPE_TO_ENTITY:
            symptoms[key] = []
        norm_chr = ""
        norm_str = ""
        for i in range(length):
            if BIO_label[i] == 'O':
                if norm_chr != "" and norm_str != "" and norm_str not in symptoms[norm_chr]:
                    symptoms[norm_chr].append(norm_str)
                norm_chr = ""
                norm_str = ""
            else:
                if norm_chr != "" and norm_chr != ENTITY_MAP[BIO_label[i]]:
                    if norm_str not in symptoms[norm_chr]:
                        symptoms[norm_chr].append(norm_str)
                    norm_chr = ENTITY_MAP[BIO_label[i]]
                    norm_str = sentence[i]
                else:
                    norm_chr = ENTITY_MAP[BIO_label[i]]
                    norm_str = norm_str + sentence[i]

        if norm_chr != "" and norm_str != "":
            symptoms[norm_chr].append(norm_str)

        return symptoms

    def pattern_handler(item):
        sentence = item["sentence"]
        BIO_label = item["BIO_label"].split(" ")

        # get map
        symptoms = get_symptoms_map(sentence, BIO_label)
        answer_choices = []
        for key in symptoms:
            if len(symptoms[key]) > 0:
                answer_choices.append(key)
        if len(answer_choices) == 0:
            return None
        symptom_list = []
        if random.random() < P_EXIST or len(answer_choices) == len(TYPE_TO_ENTITY):
            if random.random() < P_NO_ANS and len(answer_choices) < len(TYPE_TO_ENTITY):
                chosen_element = gen_diff_item(answer_choices, TYPE_TO_ENTITY, random.randint(1, len(TYPE_TO_ENTITY) - len(answer_choices)))
                if chosen_element is not None:
                    answer_choices.extend(chosen_element)

            for symptom_name in answer_choices:
                symptom_norm = "，".join(symptoms[symptom_name])
                symptom_list.append(f"{symptom_name}实体：{symptom_norm}")
            target_text = "上述句子中的实体包含：\n" + "\n".join(symptom_list)
        else:
            answer_choices = gen_diff_item(answer_choices, TYPE_TO_ENTITY, random.randint(1, len(TYPE_TO_ENTITY) - len(answer_choices)))
            target_text = "上述句子没有指定类型实体"

        key, _ = random.choice(list(PATTERN_KEY.items()))
        input_text = PATTERN_KEY[key](sentence, "，".join(answer_choices))
        if random.random() < P_ADD_ANS:
            input_text = input_text + "\n答："
        processed_data = {
            "input": input_text,
            "target": target_text,
            "answer_choices": answer_choices,
            "task_type": "ner",
            "task_dataset": "IMCS-V2-NER",
            "sample_id": "bgi"
        }
        return processed_data
    train_sample = []
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for item in obj:
            for dialogue in obj[item]["dialogue"]:
                sample = pattern_handler(dialogue)
                if sample is not None:
                    train_sample.append(sample)
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output_train_sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def P_MedDG(num_sample = 10000):
    set_random_seed(SEED)

    # 1、60.5%的概率加上'\n答：'
    # 2、12%的概率生成，'上述句子没有指定类型实体'问题，即设置一个或多个不存在的实体，则回答中全为空
    # 3、88%的问题中，要有50%的数据，自动补充一个多的不存在答案实体，回答中也只回答空

    FILE_ORIGIN_TRAIN=FILE_ORIGIN_TRAIN_MAP["MedDG"]
    FILE_PROMPT_ORIGIN_TRAIN=FILE_PROMPT_ORIGIN_TRAIN_MAP["MedDG"]
    FILE_BGI_PROMPT_TRAIN=FILE_BGI_PROMPT_TRAIN_MAP["MedDG"].replace("{NUM_SAMPLE}", str(num_sample))

    P_ADD_ANS = 0.6

    def pattern_1(conversation):
        return f"{conversation}\n根据上述对话历史，给出医生的下一句话"
    def pattern_2(conversation):
        return f"{conversation}\n根据上述对话历史，作为医生应该如何回复？"
    def pattern_3(conversation):
        return f"根据下面的问诊对话历史，给出医生的下一句回复\n{conversation}"
    def pattern_4(conversation):
        return f"自动生成问诊对话中的医生下一句回复：\n{conversation}"
    def pattern_5(conversation):
        return f"根据医生和患者交流的对话历史预测出医生的下一句回复：\n{conversation}"
    PATTERN_KEY = {
        '(.+)\n根据上述对话历史，给出医生的下一句话': pattern_1,
        '(.+)\n根据上述对话历史，作为医生应该如何回复？': pattern_2,
        '根据下面的问诊对话历史，给出医生的下一句回复\n(.+)': pattern_3,
        '自动生成问诊对话中的医生下一句回复：\n(.+)': pattern_4,
        '根据医生和患者交流的对话历史预测出医生的下一句回复：\n(.+)': pattern_5,
    }

    check_new_pattern(FILE_PROMPT_ORIGIN_TRAIN, PATTERN_KEY)

    # 统计原始数据和PromptBLUE训练数据的对话轮数
    def cal_conversations():
        from collections import Counter
        from itertools import groupby

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        o_number_docter = []
        o_number_patient = []
        with open(FILE_ORIGIN_TRAIN, 'r') as f:
            obj = json.load(f)
            for conversation in obj:
                id_list = [item['id'] for item in conversation]
                group_id = [key for key, group in groupby(id_list)]
                o_number_docter.append(group_id.count('Patient'))
                o_number_patient.append(group_id.count('Doctor'))

        p_number_docter = []
        p_number_patient = []
        with open(FILE_PROMPT_ORIGIN_TRAIN, 'r') as f:
            obj = json.load(f)
            for item in obj:
                input = item['input']
                p_number_docter.append(input.count("医生："))
                p_number_patient.append(input.count("患者："))

        p_counts_v1, p_counts_v2 = Counter(p_number_docter), Counter(p_number_patient)
        p_counts_v1, p_counts_v2 = sorted(p_counts_v1.items()),sorted(p_counts_v2.items())
        p_nums_v1, p_freq_v1 = zip(*p_counts_v1)
        p_nums_v2, p_freq_v2 = zip(*p_counts_v2)


        o_counts_v1, o_counts_v2 = Counter(o_number_docter), Counter(o_number_patient)
        o_counts_v1, o_counts_v2 = sorted(o_counts_v1.items()),sorted(o_counts_v2.items())
        o_nums_v1, o_freq_v1 = zip(*o_counts_v1)
        o_nums_v2, o_freq_v2 = zip(*o_counts_v2)
        plt.figure(figsize=(10,5))

        plt.subplot(2, 2, 1)
        plt.bar(o_nums_v1, o_freq_v1)
        plt.title('O_Doctor')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.subplot(2, 2, 2)
        plt.bar(o_nums_v2, o_freq_v2)
        plt.title('O_Patient')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.subplot(2, 2, 3)
        plt.bar(p_nums_v1, p_freq_v1)
        plt.title('P_Doctor')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.subplot(2, 2, 4)
        plt.bar(p_nums_v2, p_freq_v2)
        plt.title('P_Patient')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.show()


    def get_conversations_list(conversation) -> Tuple[List[str], List[str]]:
        docter_dialogues = []
        patient_dialogues = []
        docter_str = []
        patient_str = []
        last_persion = ""
        for item in conversation:
            if item['id'] == 'Doctor':
                docter_str.append(item['Sentence'])
                if last_persion == 'Patient':
                    patient_dialogues.append('\n'.join(patient_str))
                    patient_str.clear()
                last_persion = 'Doctor'
            elif item['id'] == 'Patient':
                patient_str.append(item['Sentence'])
                if last_persion == 'Doctor':
                    docter_dialogues.append("\n".join(docter_str))
                    docter_str.clear()
                last_persion = 'Patient'
            else:
                raise ValueError("id error")
        if docter_str:
            docter_dialogues.append("\n".join(docter_str))
        if patient_str:
            patient_dialogues.append("\n".join(patient_str))
        return docter_dialogues, patient_dialogues

    def pattern_handler(conversation):
        # 只有两个数据第一个开口的居然是医生，单独处理太麻烦，索性直接忽略
        if conversation[0]['id'] == 'Doctor':
            return None
        assert(conversation[0]['id'] == 'Patient')

        # 尝试用正态分布生成对话轮数
        d_dialogues, p_dialogues = get_conversations_list(conversation)
        normal = np.abs(np.random.normal(0, 1))
        max_counter = min(len(d_dialogues), len(p_dialogues)) - 1
        counter = min(max_counter, int(normal * max_counter))

        samples = []
        for counter in range(max_counter + 1):
            input_text = ""
            target_text = ""
            for i in range(0, counter):
                input_text += "\n\n患者：" + p_dialogues[i] + "\n\n医生：" + d_dialogues[i]
            input_text += "\n\n患者：" + p_dialogues[counter]
            target_text = d_dialogues[counter]
            input_text = input_text.lstrip("\n")

            key, _ = random.choice(list(PATTERN_KEY.items()))
            input_text = PATTERN_KEY[key](input_text)

            if random.random() < P_ADD_ANS:
                input_text = input_text + "\n答："

            processed_data = {
                "input": input_text,
                "target": target_text,
                "answer_choices": None,
                "task_type": "response_generation",
                "task_dataset": "MedDG",
                "sample_id": "bgi"
            }
            samples.append(processed_data)
        return samples
    # cal_conversations()
    train_sample = []
    with open(FILE_ORIGIN_TRAIN, "r", encoding="utf-8") as f:
        obj = json.load(f)
        for conversation in obj:
            sample = pattern_handler(conversation)
            if sample is not None:
                train_sample.extend(sample)
    print(f"all train sample: {len(train_sample)}")
    output_train_sample = random.sample(train_sample, k=num_sample)

    print(f"output train sample: {len(output_train_sample)}")
    with open(FILE_BGI_PROMPT_TRAIN, "w", encoding="utf-8") as f:
        for sample in output_train_sample:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def build_train_data():
    CHIP_CDEE()
    CMeEE_V2()
    CMeIE()
    KUAKE_QIC()
    KUAKE_QTR()
    IMCS_V2_NER()
    CHIP_CTC()
    CHIP_CDN()
    # CHIP_MDCFNPC() # not finished
    CHIP_STS()
    KUAKE_IR()
    P_MedDG()

    
    shutil.copy(PROMPTCBLUE_FILE, BGI_DATA)
    cal_file_hash(BGI_DATA, 'sha256.chk')

if __name__ == "__main__":
    split()
    build_train_data()
