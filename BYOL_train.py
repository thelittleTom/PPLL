import torch
from torch.utils.data import DataLoader
import math
from mysentence_transformers import losses
from mysentence_transformers import LoggingHandler, InputExample
from mysentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import os
import random
from argparse import ArgumentParser
import numpy as np
import json
import pickle
from InstructorEmbedding.instructor_byol import INSTRUCTOR


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_jsonl(path):
    import jsonlines
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%m-%d %H:%M',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

parser = ArgumentParser()
parser.add_argument("--data_path", default='./datasets/clinc/small_p.json', type=str,
                    help="the path of the dataset")
parser.add_argument("--save_dir", default='./output/clinc/checkpoints')
parser.add_argument("--train_batch_size", default=28, type=int)
parser.add_argument('--suffix', default='clinc_small', type=str)
parser.add_argument('--positive_pairs', default='./output/clinc/min_rsts_clinc_small.pkl', type=str)
parser.add_argument('--mini_cluster_labels', default='./output/clinc/new2_clinc_small.pkl', type=str)
parser.add_argument('--kmeans_pairs', default='./output/clinc/kmeans_pairs2_clinc_small.pkl',
                    type=str)
parser.add_argument("--model_name", default='/root/autodl-tmp/LLM-Research/instructor_large', type=str)
parser.add_argument("--save_check", default=True, type=bool)
args = parser.parse_args()
data_path = args.data_path
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

path_part = data_path.split('/')
with open('./prompt/prompts.json', 'r', encoding='utf-8') as file:
    instruction_prompt = json.load(file)
instruction_ = instruction_prompt[path_part[-2]]

data = read_jsonl(data_path)
texts = [d['text'] for d in data]
text2id = {}
for i, text in enumerate(texts):
    if text not in text2id:
        text2id[text] = i
bests = []
for lr in [1e-4, 5e-5]:  # [5e-5,1e-4]

    seed_everything(42)

    model = INSTRUCTOR(args.model_name).cuda()
    print(f" lr={lr} ")

    moving_average_decay = 0.999

    mode = 0
    dev_samples = []
    train_samples = []

    save_check = args.save_check
    with open(args.positive_pairs, "rb") as file:

        min_rsts = pickle.load(file)

    with open(args.kmeans_pairs, "rb") as file:

        eval_kmeans_samples = pickle.load(file)


    check_path = f'{args.save_dir}/{lr}'


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
                                  drop_last=True)

    train_loss = losses.BYOLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                moving_average_decay=moving_average_decay, mode=mode)

    warmup_steps = math.ceil(len(train_dataloader) * 0.1)
    evaluation_steps = int(len(train_dataloader) * 0.25)

    bests.append(model.fit(train_objectives=[(train_dataloader, train_loss)],
                           evaluator=dev_evaluator,
                           epochs=1,
                           evaluation_steps=evaluation_steps,
                           warmup_steps=warmup_steps,
                           output_path=args.save_dir,
                           optimizer_params={'lr': lr},
                           max_grad_norm=1,
                           mode=mode,
                           checkpoint_path=check_path if save_check else None,
                           checkpoint_save_steps=evaluation_steps,
                           use_amp=False
                           )
                 )
    bests[-1]['best_pth'] = check_path + '/' + bests[-1]['best_pth']
print('----the best checkpoint----')
print(bests[0] if bests[0]['score'] > bests[1]['score'] else bests[1])
