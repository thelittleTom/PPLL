from sklearn.metrics import normalized_mutual_info_score

import numpy as np

import jsonlines
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment as linear_assignment

from InstructorEmbedding import INSTRUCTOR
from InstructorEmbedding import INSTRUCTOR
import torch
from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument("--data_path", default='./datasets/massive_intent/small_p.json', type=str,
                    help="the path of the dataset")
parser.add_argument("--save_dir", default='./output/massive_intent')
parser.add_argument('--suffix', default='massive_intent_small', type=str)
parser.add_argument('--positive_pairs', default='./output/massive_intent/min_rsts_massive_intent_small.pkl', type=str)
parser.add_argument('--mini_cluster_labels', default='./output/massive_intent/new2_massive_intent_small.pkl', type=str)

parser.add_argument("--model_name", default='/root/autodl-tmp/LLM-Research/instructor_large', type=str)
parser.add_argument("--model_pth",default='./output/massive_intent/checkpoints/0.0001/0_186.pth',type=str)
parser.add_argument("--eval_add_labels", default=False, type=bool)
parser.add_argument("--K_", default=59*3, type=int)
args = parser.parse_args()
data_path = args.data_path
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

model_embed = INSTRUCTOR(args.model_name).cuda()
state_dict = torch.load(args.model_pth)
model_embed.load_state_dict(state_dict)



def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


path_part = data_path.split('/')
K_ = args.K_
data = read_jsonl(data_path)
belta = len(data) / K_
import pickle
with open(args.positive_pairs, "rb") as file:
    min_rsts = pickle.load(file)

min_rsts_scatter = [mr for mrr in min_rsts for mr in mrr]
from collections import Counter
from sklearn.cluster import KMeans

list_llm = [_['label_llm'] for _ in min_rsts_scatter]
count_dict = Counter(list_llm)

new_list_llm = []

for item, count in count_dict.most_common():
    print(f"{item}: {count}")
    new_list_llm.append(item)
print(f"new list llm  {len(new_list_llm)}")

from InstructorEmbedding import INSTRUCTOR

state_dict = torch.load(args.model_pth)
model_embed.load_state_dict(state_dict)

import json


with open('./prompt/prompts.json', 'r', encoding='utf-8') as file:
    instruction_prompt = json.load(file)
instruction_ = instruction_prompt[path_part[-2]]

batch_size = 2000
list_sen_embedding = []


cur_use_data = list_llm

for i in tqdm(range(0, len(cur_use_data), batch_size)):
    batch = cur_use_data[i:i + batch_size]
    try:
        sentences = [[instruction_, item['text']] for item in batch]
    except:
        sentences = [[instruction_, item] for item in batch]
    sentence_vectors = model_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    list_sen_embedding.extend(sentence_vectors)

list_data_embedding = list_sen_embedding

estimate_k = []
from sklearn.cluster import KMeans

for seed in [100, 13, 21, 36, 42]:
    kmeans = KMeans(n_clusters=K_, random_state=seed, )

    # kmeans = KMeans(n_clusters=77, max_iter=10000, random_state=42, init=init_c,, init='k-means++', )
    kmeans.fit(list_data_embedding)

    cluster_labels = kmeans.labels_
    count_label = Counter(cluster_labels)
    all_llm_labels = []

    belta = len(cur_use_data) / K_
    for item, count in count_label.most_common():
        print(f"{item}: {count}")
        if count > belta:
            all_llm_labels.append(item)
    print(len(all_llm_labels))
    estimate_k.append(len(all_llm_labels))
print(np.mean(estimate_k))