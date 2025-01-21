import torch
from torch.utils.data import DataLoader  # 用于加载数据
import math
from mysentence_transformers import   losses  # 引入模型和损失函数
from mysentence_transformers import LoggingHandler,   InputExample  # 引入日志、模型、工具和输入样本
from mysentence_transformers.evaluation import EmbeddingSimilarityEvaluator  # 引入评估器
import logging
import os
import random

import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保结果可重复
    torch.backends.cudnn.benchmark = True  # 优化卷积网络性能


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%m-%d %H:%M',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_path", default='./datasets/massive_intent/small_p.json', type=str,
                    help="the path of the dataset")
parser.add_argument("--save_dir", default='./output/massive_intent/checkpoints')
parser.add_argument("--train_batch_size", default=28, type=int)
parser.add_argument('--suffix', default='massive_intent_small', type=str)
parser.add_argument('--positive_pairs', default='./output/massive_intent/min_rsts_massive_intent_small.pkl', type=str)
parser.add_argument('--mini_cluster_labels', default='./output/massive_intent/new2_massive_intent_small.pkl', type=str)
parser.add_argument('--kmeans_pairs', default='./output/massive_intent/kmeans_pairs2_massive_intent_small.pkl',
                    type=str)
parser.add_argument("--model_name", default='/root/autodl-tmp/LLM-Research/instructor_large', type=str)

args = parser.parse_args()
data_path = args.data_path
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

import json

path_part = data_path.split('/')
with open('./prompt/prompts.json', 'r', encoding='utf-8') as file:
    instruction_prompt = json.load(file)
instruction_ = instruction_prompt[path_part[-2]]


def read_jsonl(path):
    import jsonlines
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


data = read_jsonl(data_path)
texts = [d['text'] for d in data]
text2id = {}
for i, text in enumerate(texts):
    if text not in text2id:
        text2id[text] = i
bests=[]
for lr in [1e-4, 5e-5]:  # [5e-5,1e-4]

    seed_everything(42)
    from InstructorEmbedding.instructor_byol import INSTRUCTOR

    model = INSTRUCTOR(args.model_name).cuda()
    print(f" lr={lr} ")

    moving_average_decay = 0.999  # 移动平均衰减率

    mode = 0
    train_samples = []
    dev_samples = []
    import pickle

    n_clusters = 11

    save_check = True
    with open(args.positive_pairs, "rb") as file:
        # 使用 pickle.load() 恢复嵌套列表
        min_rsts = pickle.load(file)

    with open(args.kmeans_pairs, "rb") as file:
        # 使用 pickle.load() 恢复嵌套列表
        eval_kmeans_samples = pickle.load(file)

    train_samples = []

    if save_check:
        check_path = f'{args.save_dir}/{lr}'
    else:
        check_path = None

    min_rsts_scatter = [mr for mrr in min_rsts for mr in mrr]
    prediction = {}
    for m in min_rsts_scatter:
        if m['label_llm'] not in prediction:
            prediction[m['label_llm']] = []
        prediction[m['label_llm']].append(m)

    for k, v in prediction.items():
        for i, vi in enumerate(v):
            for j in range(i, len(v)):
                if i != j:
                    train_samples.append(InputExample(texts=[[instruction_, vi['text']], [instruction_, v[j]['text']]],
                                                      label=1))

    dev_evaluator = EmbeddingSimilarityEvaluator(eval_kmeans_samples=eval_kmeans_samples,
                                                 data_path=data_path, instruction_=instruction_,
                                                 mini_cluster_labels=args.mini_cluster_labels)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size,
                                  drop_last=True)  # 加载训练数据

    train_loss = losses.BYOLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                moving_average_decay=moving_average_decay, mode=mode)  # 使用BYOLoss作为损失函数

    # 设置warmup步数和评估步数
    warmup_steps = math.ceil(len(train_dataloader) * 0.1)
    evaluation_steps = int(len(train_dataloader) * 0.25)
    logging.info("训练句子数量: {}".format(len(train_samples)))
    logging.info("Warmup步骤数: {}".format(warmup_steps))

    # 开始训练模型
    bests.append(model.fit(train_objectives=[(train_dataloader, train_loss)],  # 训练目标
              evaluator=dev_evaluator,  # 评估器
              epochs=1,  # 训练轮数
              evaluation_steps=evaluation_steps,  # 评估步数
              warmup_steps=warmup_steps,  # warmup步数
              output_path=args.save_dir,  # 输出路径
              optimizer_params={'lr': lr},  # 优化器参数
              max_grad_norm=1,
              mode=mode,
              checkpoint_path=check_path,
              checkpoint_save_steps=evaluation_steps,
              use_amp=False  # 如果GPU支持FP16运算，则使用混合精度训练
              )
                 )
    bests[-1]['best_pth']=check_path+'/'+bests[-1]['best_pth']
print(bests[0] if bests[0]['score'] > bests[1]['score'] else bests[1])