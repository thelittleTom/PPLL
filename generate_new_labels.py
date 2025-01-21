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
args = parser.parse_args()
data_path = args.data_path
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

model_embed = INSTRUCTOR(args.model_name).cuda()
state_dict = torch.load(args.model_pth)
model_embed.load_state_dict(state_dict)

import pickle

import json
def read_jsonl(path):
    import jsonlines
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content
path_part = data_path.split('/')
with open('./prompt/prompts.json', 'r', encoding='utf-8') as file:
    instruction_prompt = json.load(file)
instruction_ = instruction_prompt[path_part[-2]]
model_embed.eval()

data = read_jsonl(data_path)
true_labels = [item['label'] for item in data]
true_label_num = len(set(true_labels))
label2text = {}
text2id = {}
text2data = {}
for i, d in enumerate(data):
    text2id[d['text']] = i
    d['llm'] = []
    text2data[d['text']] = d

    if d['label'] not in label2text:
        label2text[d['label']] = d['label_text']
if args.eval_add_labels:

    # with open(f"/root/autodl-tmp/paper_cluster/finetuning/embed_finetuning/data/new2_{suffix}.pkl", "wb") as file:
    #     pickle.dump(new_labels2, file)
    with open(args.mini_cluster_labels,
              "rb") as file:
        # 使用 pickle.load() 恢复嵌套列表
        labels_v4 = pickle.load(file)

    fake_n_data = []
    for l in labels_v4:
        fake_n_data.append({'text': l})
    data = data + fake_n_data
with open( args.positive_pairs, "rb") as file:
    # 使用 pickle.load() 恢复嵌套列表
    min_rsts = pickle.load(file)

min_rsts_scatter = [mr for mrr in min_rsts for mr in mrr]
for m in min_rsts_scatter:
    text2data[m['text']]['llm'].append(m['label_llm'])
test = [[instruction_, item['text']] for item in data]
batch_size = 2000
list_sen_embedding = []
# 将数据按batch_size分块
for i in  range(0, len(data), batch_size) :
    batch = data[i:i + batch_size]

    sentences = [[instruction_, item['text']] for item in batch]

    sentence_vectors = model_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    # 存储句子向量和原始数据
    list_sen_embedding.extend(sentence_vectors)

list_data_embedding = list_sen_embedding

label2text_set = {}
for d in data:
    if d['label'] not in label2text_set:
        label2text_set[d['label']] = d['label_text']
all_acc = []
all_NMI = []
all_score = []
all_sss = []
all_ss = []
all_mo = []
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score, silhouette_samples
def calculate_acc(y_true, y_pred,texts=None):
    """
    计算聚类的准确性 (ACC)
    y_true: 真实标签，形状为 (n_samples,)
    y_pred: 聚类标签，形状为 (n_samples,)
    返回值：聚类准确性 (ACC)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if texts is not None:
        texts_true, texts_pred = texts
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        cost_matrix[y_pred[i], y_true[i]] += 1

    idx1, idx2 = linear_assignment(cost_matrix.max() - cost_matrix)

    total_sum = sum([cost_matrix[i, j] for i, j in zip(idx1, idx2)])

    mapping = {}
    mapping_dict={}
    for i1, i2 in zip(idx1, idx2):
        mapping[i1] = i2
        if texts is not None:
            print(f"{texts_pred[i1]}-----> {texts_true[i2]}")



    mapped_arr = np.array([mapping.get(x, x) for x in y_pred])

    return total_sum / y_pred.size, mapped_arr
for seed in [42]:
    kmeans = KMeans(n_clusters=true_label_num, random_state=seed, )

    kmeans.fit(list_data_embedding)
    cluster_labels = kmeans.labels_[:len(true_labels)]
    acc, new_cluster_labels = calculate_acc(true_labels, cluster_labels)

    for one, cl in zip(data, new_cluster_labels[: len(data) ]):
        one['predict'] = int(cl)
    preds = {}
    for d in data:
        if d['predict'] not in preds:
            preds[d['predict']] = {}
            preds[d['predict']]['data'] = []
            preds[d['predict']]['llm'] = []
        preds[d['predict']]['data'].append(d)
        preds[d['predict']]['llm'].append(text2data[d['text']]['llm'])
    for k, v in preds.items():
        print('pred', k, label2text_set[k])
        list_llm = [__ for _ in preds[k]['llm'] for __ in _]
        count_dict = Counter(list_llm)

        # 打印每个元素及其数量
        for item, count in count_dict.most_common(10):
            print(f"{item}: {count}")
prompt1 = """\
Instruction
##Context
- *Goal*  Your goal is to summary the input data into a meaningful LABEL **{according}**.
- *Data*  The input data will be a markdown table containing category descriptions and the corresponding number of utterances for each category, with the following columns:
  - **category description**   A description of the category, generated by clustering related utterances.
  - **number**  The number of utterances that belong to this category.

##Requirements
- Provide your answers between the tags: <summary>your generated summary LABEL with less than 8 words</summary>, <explanation>explanation of your reasoning process within {n_exp} words</explanation>.

# Data
{data_table}

# Output
"""
from llm_utils import completion_with_backoff
llm_labels=[]
according = 'according to the intent'
for k, v in preds.items():

    list_llm = [__ for _ in preds[k]['llm'] for __ in _]
    count_dict = Counter(list_llm)

    # 打印每个元素及其数量

    a = """|category description|number|\n"""
    label_ = []

    for item, count in count_dict.most_common(3):  # enumerate(preds_nopred[pren]):
        label_.append(f"|{item}|{count}|")

    print(a + '\n'.join(label_))
    input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=250, according=according,
                                  )

    response = completion_with_backoff(input_prompt,
                                       llm='llm_35_stable')

    try:
        response = response.split('<summary>')[-1].split('</summary>')[0].strip()
        print(label2text_set[k], '--->summary:', response)
        preds[k]['pred_summary'] = response
        preds[k]['pred_label_text'] = label2text_set[k]
        llm_labels.append(response)
    except:
        continue


# 将列表的每个元素写入到txt文件，每个元素占一行
with open(args.save_dir+'/new_labels.txt', 'w') as file:
    for item in llm_labels:
        file.write(item + '\n')

