from llm_utils import completion_with_backoff
import random
import torch
import jsonlines
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import AgglomerativeClustering
from dotmap import DotMap
import numpy as np
from sklearn.metrics import silhouette_score
import re


def write_jsonl(path, content):
    with jsonlines.open(path, "w") as json_file:
        json_file.write_all(content)


def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def get_label(text, n_data):
    for d in n_data:
        if ''.join(d['text'].split()) == ''.join(text.split()):
            return d['label'], d['label_text']

    return None, None


def extract_digits(s):
    pattern = r'^(\d+)-(\d+)$'
    match = re.match(pattern, s)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None


def llm_label_v4(config_pre, data, model_embed, according='according to the intent',
                 target_text='intent', prompt1=None, llm='gpt3', llm_model=None, entropy_num=25):
    model_embed.eval()

    def entropy(vals):
        vals = np.asarray(vals)
        vals /= vals.sum()
        return - (vals * np.log(vals)).sum()

    intent_set = set()
    for d in data:
        intent_set.add(d['label_text'])

    n_clusters = config_pre.n_classes

    batch_size = 5000
    list_sen_embedding = []

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]

        sentences = [[instruction_, item['text']] for item in batch]

        sentence_vectors = model_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

        list_sen_embedding.extend(sentence_vectors)
    from sklearn.cluster import KMeans
    ##print(f'start cluster {n_clusters}')

    kmeans = KMeans(n_clusters=n_clusters, random_state=config_pre.seed)
    kmeans.fit(list_sen_embedding)
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    for one, cl in zip(data, cluster_labels):
        one['predict'] = int(cl)
    preds = {}
    for d in data:
        if d['predict'] not in preds:
            preds[d['predict']] = []
        preds[d['predict']].append(d)
    list_sen_embedding = np.array(list_sen_embedding)

    entropies = []
    options = []

    cl_sims = []
    for idx in range(len(list_sen_embedding)):
        dist = ((list_sen_embedding[idx] - centers) ** 2).sum(-1)
        prob = (1 + dist) ** (-1)
        prob /= prob.sum()
        cl_sims.append(prob[cluster_labels[idx]])  # cluster similarity scores
        # select most probable clusters
        sorted_prob = np.argsort(prob)[::-1][:entropy_num]

        options.append(sorted_prob)  # most probable cluster index
        entropies.append(entropy(prob[sorted_prob]))  #

    for one, cl, ds, embed in zip(data, cluster_labels, entropies, list_sen_embedding):
        one['predict'] = int(cl)
        one['distance'] = ds
        one['embedding'] = embed

    sort_metric = 'distance'
    data = sorted(data, key=lambda x: x[sort_metric])

    labels_group_all = []
    if_sequence = False

    for i, predclass in enumerate(preds.keys()):
        label_groups = []
        pcluster = sorted(preds[predclass], key=lambda x: x[sort_metric])

        closes = sorted(pcluster, key=lambda x: x[sort_metric])
        split_n = int(len(closes) / config_pre.nc)
        if split_n == 0:
            p_cluster_texts = [c['text'] for c in closes]
            label_groups.append(p_cluster_texts)
            labels_group_all.append(label_groups)
            continue
        half = int(config_pre.nc / 2)
        if len(closes) % config_pre.nc <= half:
            if config_pre.all_data_v4:
                pass
            else:
                closes = closes[:split_n * config_pre.nc]
        pad = 0
        for sp in range(split_n):
            # a = closes[sp * half:(sp + 1) * half]
            # b = closes[len(closes) - (sp + 1) * half:len(closes) - sp * half]

            if if_sequence:
                cc = closes[sp * (config_pre.nc + pad):(sp + 1) * (config_pre.nc + pad)]
            else:
                a = closes[sp * (half + pad):(sp + 1) * (half + pad)]
                b = closes[len(closes) - (sp + 1) * (half - pad):len(closes) - sp * (half - pad)]
                cc = a + b
            p_cluster_texts = [c['text'] for c in cc]
            label_groups.append(p_cluster_texts)
        if len(closes) % config_pre.nc != 0:
            # p_cluster_texts = [c['text'] for c in closes[split_n * half:len(closes) - split_n * half]]
            if if_sequence:
                p_cluster_texts = [c['text'] for c in
                                   closes[split_n * (config_pre.nc + pad):]]
            else:
                p_cluster_texts = [c['text'] for c in
                                   closes[split_n * (half + pad):len(closes) - split_n * (half - pad)]]
            if config_pre.all_data_v4:
                if len(p_cluster_texts) >= int(half * 2 / 3):
                    label_groups.append(p_cluster_texts)
                else:
                    label_groups[-1].extend(p_cluster_texts)
            else:
                label_groups.append(p_cluster_texts)
        labels_group_all.append(label_groups)

    label_ = [l for lll in labels_group_all for ll in lll for l in ll]
    ##print('label_======', len(label_))

    rst = []
    min_rsts = []

    for lab_idx, labels_groups in tqdm(enumerate(labels_group_all)):

        for li, lab in enumerate(labels_groups):

            a = """|id|utterance|\n"""
            label_ = []
            a_dict = {}
            for i, l in enumerate(lab):  # enumerate(preds_nopred[pren]):
                label_.append(f"|{i + 1}|{l}|")
                a_dict[i + 1] = l

            ##print(a + '\n'.join(label_))
            ##print('merge--------------')
            if llm == 'gpt4o':
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=50, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llm_4o_stable')
            elif llm == 'gpt3':
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=250, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llm_35_stable')
            elif 'gpt' not in llm:
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=250, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llama3', model=llm_model[0], tokenizer=llm_model[1])
            # ,,llm='llm_35_stable'
            #print(response)

            try:
                response = response.replace('```', '')
                response = response.split('<cluster table>')[-1].split('</cluster table>')[0].strip().split('\n')
                if 'markdown' in response[0]:
                    response = response[1:]
                if '---' in response[1]:
                    response = response[2:]
                else:
                    response = response[1:]
            except:
                continue
            min_rst = []

            for r in response:
                try:
                    cur_label = r.split('|')[2].strip()
                except:
                    #print("error happen in r.split('|')[2].strip()", r)
                    continue
                extract_di = extract_digits(r.split('|')[1].strip())
                if extract_di is not None:
                    r_idxs = list(range(extract_di[0], extract_di[1]))
                else:
                    r_idxs = r.split('|')[1].strip().split(',')

                new_texts = []
                for r_idx in r_idxs:
                    try:
                        ridx = int(r_idx)
                        new_texts.append(a_dict[ridx])
                    except:
                        continue

                    label_idx, label_name = get_label(a_dict[ridx], data)
                    rst.append({'text': a_dict[ridx], 'label_llm': cur_label, 'org_cluster': (lab_idx, li, ridx),
                                'ground_label': (label_idx, label_name)})
                    min_rst.append({'text': a_dict[ridx], 'label_llm': cur_label, 'org_cluster': (lab_idx, li, ridx),
                                    'ground_label': (label_idx, label_name)})
                    #print(ridx, a_dict[ridx], (label_idx, label_name))

            if len(min_rst) != len(lab):
                #print("not enough labels")
                labels_groups.append(lab[:int(len(lab) / 2)])
                labels_groups.append(lab[int(len(lab) / 2):])
                continue

            min_rsts.append(min_rst)

    return rst, min_rsts


def llm_label_v4_random(config_pre, data, model_embed, according='according to the intent',
                        target_text='intent', prompt1=None, llm='gpt3', llm_model=None, entropy_num=25):
    model_embed.eval()

    intent_set = set()
    for d in data:
        intent_set.add(d['text'])
    org_cluster = len(intent_set)
    n_clusters = org_cluster  # int(len(data)/60)
    if config_pre.n_classes:
        n_clusters = config_pre.n_classes

    batch_size = 5
    list_sen_embedding = []

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]

        sentences = [[instruction_, item['text']] for item in batch]

        sentence_vectors = model_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

        list_sen_embedding.extend(sentence_vectors)
    from sklearn.cluster import KMeans
    #print(f'start cluster {n_clusters}')

    kmeans = KMeans(n_clusters=n_clusters, random_state=config_pre.seed)
    kmeans.fit(list_sen_embedding)
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    for one, cl in zip(data, cluster_labels):
        one['predict'] = int(cl)
    preds = {}
    for d in data:
        if d['predict'] not in preds:
            preds[d['predict']] = []
        preds[d['predict']].append(d)
    list_sen_embedding = np.array(list_sen_embedding)

    for one, cl, embed in zip(data, cluster_labels, list_sen_embedding):
        one['predict'] = int(cl)
        one['embedding'] = embed

    labels_group_all = []
    if_sequence = False

    import random
    random.seed(42)
    for i, predclass in enumerate(preds.keys()):
        label_groups = []
        closes = preds[predclass]
        # pcluster = sorted(preds[predclass], key=lambda x: x[sort_metric])
        random.shuffle(closes)
        split_n = int(len(closes) / config_pre.nc)
        if split_n == 0:
            p_cluster_texts = [c['text'] for c in closes]
            label_groups.append(p_cluster_texts)
            labels_group_all.append(label_groups)
            continue
        half = int(config_pre.nc / 2)
        if len(closes) % config_pre.nc <= half:
            if config_pre.all_data_v4:
                pass
            else:
                closes = closes[:split_n * config_pre.nc]
        pad = 0
        for sp in range(split_n):
            # a = closes[sp * half:(sp + 1) * half]
            # b = closes[len(closes) - (sp + 1) * half:len(closes) - sp * half]

            if if_sequence:
                cc = closes[sp * (config_pre.nc + pad):(sp + 1) * (config_pre.nc + pad)]
            else:
                a = closes[sp * (half + pad):(sp + 1) * (half + pad)]
                b = closes[len(closes) - (sp + 1) * (half - pad):len(closes) - sp * (half - pad)]
                cc = a + b
            p_cluster_texts = [c['text'] for c in cc]
            label_groups.append(p_cluster_texts)
        if len(closes) % config_pre.nc != 0:
            # p_cluster_texts = [c['text'] for c in closes[split_n * half:len(closes) - split_n * half]]
            if if_sequence:
                p_cluster_texts = [c['text'] for c in
                                   closes[split_n * (config_pre.nc + pad):]]
            else:
                p_cluster_texts = [c['text'] for c in
                                   closes[split_n * (half + pad):len(closes) - split_n * (half - pad)]]
            if config_pre.all_data_v4:
                if len(p_cluster_texts) >= int(half * 2 / 3):
                    label_groups.append(p_cluster_texts)
                else:
                    label_groups[-1].extend(p_cluster_texts)
            else:
                label_groups.append(p_cluster_texts)
        labels_group_all.append(label_groups)

    label_ = [l for lll in labels_group_all for ll in lll for l in ll]
    #print('label_======', len(label_))

    rst = []

    min_rsts = []

    for lab_idx, labels_groups in enumerate(labels_group_all):

        for li, lab in enumerate(labels_groups):

            a = """|id|utterance|\n"""
            label_ = []
            a_dict = {}
            for i, l in enumerate(lab):  # enumerate(preds_nopred[pren]):
                label_.append(f"|{i + 1}|{l}|")
                a_dict[i + 1] = l

            #print(a + '\n'.join(label_))
            #print('merge--------------')
            if llm == 'gpt4':
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=50, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llm_4_stable')
            elif llm == 'gpt3':
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=250, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llm_35_stable')
            elif 'gpt' not in llm:
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=250, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llama3', model=llm_model[0], tokenizer=llm_model[1])
            # ,,llm='llm_35_stable'
            #print(response)

            try:
                response = response.replace('```', '')
                response = response.split('<cluster table>')[-1].split('</cluster table>')[0].strip().split('\n')
                if 'markdown' in response[0]:
                    response = response[1:]
                if '---' in response[1]:
                    response = response[2:]
                else:
                    response = response[1:]
            except:
                continue
            min_rst = []
            tmp_group_main_dict = {'num': 0, 'label': None}
            for r in response:
                try:
                    cur_label = r.split('|')[2].strip()
                except:
                    continue
                extract_di = extract_digits(r.split('|')[1].strip())
                if extract_di is not None:
                    r_idxs = list(range(extract_di[0], extract_di[1]))
                else:
                    r_idxs = r.split('|')[1].strip().split(',')

                new_texts = []
                for r_idx in r_idxs:
                    try:
                        ridx = int(r_idx)
                        new_texts.append(a_dict[ridx])
                    except:
                        continue

                    label_idx, label_name = get_label(a_dict[ridx], data)
                    rst.append({'text': a_dict[ridx], 'label_llm': cur_label, 'org_cluster': (lab_idx, li, ridx),
                                'ground_label': (label_idx, label_name)})
                    min_rst.append({'text': a_dict[ridx], 'label_llm': cur_label, 'org_cluster': (lab_idx, li, ridx),
                                    'ground_label': (label_idx, label_name)})
                    #print(ridx, a_dict[ridx], (label_idx, label_name))

                if len(new_texts) > tmp_group_main_dict['num']:
                    tmp_group_main_dict['num'] = len(new_texts)
                    tmp_group_main_dict['label'] = cur_label

            if len(min_rst) != len(lab):
                #print("not enough labels")
                labels_groups.append(lab[:int(len(lab) / 2)])
                labels_groups.append(lab[int(len(lab) / 2):])
                continue

            min_rsts.append(min_rst)

    return rst, min_rsts


def llm_label_v4_random_all(config_pre, data, model_embed, according='according to the intent',
                            target_text='intent', prompt1=None, llm='gpt3', llm_model=None, entropy_num=25):
    random.shuffle(data)

    labels_group_all = [[[_['text'] for _ in data[i:i + config_pre.nc]] for i in range(0, len(data), config_pre.nc)]]

    label_ = [l for lll in labels_group_all for ll in lll for l in ll]
    #print('label_======', len(label_))

    rst = []

    min_rsts = []

    for lab_idx, labels_groups in enumerate(labels_group_all):

        for li, lab in enumerate(labels_groups):

            a = """|id|utterance|\n"""
            label_ = []
            a_dict = {}
            for i, l in enumerate(lab):  # enumerate(preds_nopred[pren]):
                label_.append(f"|{i + 1}|{l}|")
                a_dict[i + 1] = l

            #print(a + '\n'.join(label_))
            #print('merge--------------')
            if llm == 'gpt4':
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=50, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llm_4_stable')
            elif llm == 'gpt3':
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=250, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llm_35_stable')
            elif 'gpt' not in llm:
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=250, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llama3', model=llm_model[0], tokenizer=llm_model[1])
            # ,,llm='llm_35_stable'
            #print(response)

            try:
                response = response.replace('```', '')
                response = response.split('<cluster table>')[-1].split('</cluster table>')[0].strip().split('\n')
                if 'markdown' in response[0]:
                    response = response[1:]
                if '---' in response[1]:
                    response = response[2:]
                else:
                    response = response[1:]
            except:
                continue
            min_rst = []
            tmp_group_main_dict = {'num': 0, 'label': None}
            for r in response:
                try:
                    cur_label = r.split('|')[2].strip()
                except:
                    #print("r.split('|')[2].strip()", r)
                    continue
                extract_di = extract_digits(r.split('|')[1].strip())
                if extract_di is not None:
                    r_idxs = list(range(extract_di[0], extract_di[1]))
                else:
                    r_idxs = r.split('|')[1].strip().split(',')

                new_texts = []
                for r_idx in r_idxs:
                    try:
                        ridx = int(r_idx)
                        new_texts.append(a_dict[ridx])
                    except:
                        continue

                    label_idx, label_name = get_label(a_dict[ridx], data)
                    rst.append({'text': a_dict[ridx], 'label_llm': cur_label, 'org_cluster': (lab_idx, li, ridx),
                                'ground_label': (label_idx, label_name)})
                    min_rst.append({'text': a_dict[ridx], 'label_llm': cur_label, 'org_cluster': (lab_idx, li, ridx),
                                    'ground_label': (label_idx, label_name)})
                    #print(ridx, a_dict[ridx], (label_idx, label_name))

                if len(new_texts) > tmp_group_main_dict['num']:
                    tmp_group_main_dict['num'] = len(new_texts)
                    tmp_group_main_dict['label'] = cur_label

                # if flag:
                #     for r_idx in r_idxs:
                #         if len(r_idx.strip()) > 0:
                #             ridx = int(r_idx)
                #             for fnext in for_next[lab_idx][ridx - 1]:
                #                 if fnext['label_text'] == main_label_name:
                #                     main_cnt_yes += 1
                #                 else:
                #                     main_cnt_no += 1
            if len(min_rst) != len(lab):
                #print("not enough labels")
                labels_groups.append(lab[:int(len(lab) / 2)])
                labels_groups.append(lab[int(len(lab) / 2):])
                continue

            min_rsts.append(min_rst)

    return rst, min_rsts


def llm_label_v4_mini(config_pre, data, model_embed,
                      according='according to the intent', target_text='intent', prompt1=None, mini_avg=4, llm='gpt3',
                      llm_model=None):
    model_embed.eval()

    def entropy(vals):
        vals = np.asarray(vals)
        vals /= vals.sum()
        return - (vals * np.log(vals)).sum()

    intent_set = set()
    for d in data:
        intent_set.add(d['label_text'])
    org_cluster = len(intent_set)
    n_clusters = org_cluster
    if config_pre.n_classes:
        n_clusters = config_pre.n_classes

    batch_size = 10000
    list_sen_embedding = []

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]

        sentences = [[instruction_, item['text']] for item in batch]

        sentence_vectors = model_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

        list_sen_embedding.extend(sentence_vectors)
    from sklearn.cluster import KMeans
    #print(f'start cluster {n_clusters}')

    kmeans = KMeans(n_clusters=int(len(data) / mini_avg), init='k-means++', )
    kmeans.fit(list_sen_embedding)
    cluster_labels = kmeans.labels_
    from sklearn.metrics import pairwise_distances_argmin_min
    centroids = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(centroids, list_sen_embedding)
    symbolic_datas = [data[idx] for idx in closest]

    text2data = {}
    neighbor_samples = []
    for idx, sd in zip(closest, symbolic_datas):
        label_idx = cluster_labels[idx]
        neighbor = [data[_] for _ in np.where(cluster_labels == label_idx)[0]]
        data[idx]['neighbor'] = neighbor
        for i, vi in enumerate(neighbor):
            for j in range(i + 1):
                neighbor_samples.append([vi, neighbor[j]])
        text2data[sd['text']] = sd
    with open(f"/root/autodl-tmp/paper_cluster/finetuning/embed_finetuning/data/neighbor_{config_pre.suffix}.pkl",
              "wb") as file:
        pickle.dump(neighbor_samples, file)

    list_sen_embedding = [list_sen_embedding[idx] for idx in closest]
    kmeans = KMeans(n_clusters=n_clusters, random_state=config_pre.seed)
    kmeans.fit(list_sen_embedding)
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    for one, cl in zip(symbolic_datas, cluster_labels):
        one['predict'] = int(cl)
    preds = {}
    for d in symbolic_datas:
        if d['predict'] not in preds:
            preds[d['predict']] = []
        preds[d['predict']].append(d)
    list_sen_embedding = np.array(list_sen_embedding)

    entropies = []
    options = []

    cl_sims = []
    for idx in range(len(list_sen_embedding)):
        dist = ((list_sen_embedding[idx] - centers) ** 2).sum(-1)
        prob = (1 + dist) ** (-1)
        prob /= prob.sum()
        cl_sims.append(prob[cluster_labels[idx]])  # cluster similarity scores
        # select most probable clusters
        sorted_prob = np.argsort(prob)[::-1][:25]

        options.append(sorted_prob)  # most probable cluster index
        entropies.append(entropy(prob[sorted_prob]))  #

    for one, cl, ds, embed in zip(symbolic_datas, cluster_labels, entropies, list_sen_embedding):
        one['predict'] = int(cl)
        one['distance'] = ds
        one['embedding'] = embed

    sort_metric = 'distance'

    labels_group_all = []
    if_sequence = False

    for i, predclass in enumerate(preds.keys()):
        label_groups = []
        pcluster = sorted(preds[predclass], key=lambda x: x[sort_metric])

        closes = sorted(pcluster, key=lambda x: x[sort_metric])
        split_n = int(len(closes) / config_pre.nc)
        if split_n == 0:
            p_cluster_texts = [c['text'] for c in closes]
            label_groups.append(p_cluster_texts)
            labels_group_all.append(label_groups)
            continue
        half = int(config_pre.nc / 2)
        if len(closes) % config_pre.nc <= half:
            if config_pre.all_data_v4:
                pass
            else:
                closes = closes[:split_n * config_pre.nc]
        pad = 0
        for sp in range(split_n):
            # a = closes[sp * half:(sp + 1) * half]
            # b = closes[len(closes) - (sp + 1) * half:len(closes) - sp * half]

            if if_sequence:
                cc = closes[sp * (config_pre.nc + pad):(sp + 1) * (config_pre.nc + pad)]
            else:
                a = closes[sp * (half + pad):(sp + 1) * (half + pad)]
                b = closes[len(closes) - (sp + 1) * (half - pad):len(closes) - sp * (half - pad)]
                cc = a + b
            p_cluster_texts = [c['text'] for c in cc]
            label_groups.append(p_cluster_texts)
        if len(closes) % config_pre.nc != 0:
            # p_cluster_texts = [c['text'] for c in closes[split_n * half:len(closes) - split_n * half]]
            if if_sequence:
                p_cluster_texts = [c['text'] for c in
                                   closes[split_n * (config_pre.nc + pad):]]
            else:
                p_cluster_texts = [c['text'] for c in
                                   closes[split_n * (half + pad):len(closes) - split_n * (half - pad)]]
            if config_pre.all_data_v4:
                if len(p_cluster_texts) >= int(half * 2 / 3):
                    label_groups.append(p_cluster_texts)
                else:
                    label_groups[-1].extend(p_cluster_texts)
            else:
                label_groups.append(p_cluster_texts)
        labels_group_all.append(label_groups)
        a = [l for ll in label_groups for l in ll]


    label_ = [l for lll in labels_group_all for ll in lll for l in ll]
    #print('label_======', len(label_))

    rst = []
    min_rsts = []

    for lab_idx, labels_groups in enumerate(labels_group_all):

        for li, lab in enumerate(labels_groups):

            a = """|id|utterance|\n"""
            label_ = []
            a_dict = {}
            for i, l in enumerate(lab):  # enumerate(preds_nopred[pren]):
                label_.append(f"|{i + 1}|{l}|")
                a_dict[i + 1] = l

            #print(a + '\n'.join(label_))
            #print('merge--------------')
            if llm == 'gpt4':
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=50, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llm_4_stable')
            elif llm == 'gpt3':
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=250, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llm_35_stable')
            elif 'gpt' not in llm:
                input_prompt = prompt1.format(data_table=a + '\n'.join(label_), n_exp=250, according=according,
                                              target_text=target_text)

                response = completion_with_backoff(input_prompt,
                                                   llm='llama3', model=llm_model[0], tokenizer=llm_model[1])
            # ,,llm='llm_35_stable'
            #print(response)

            try:
                response = response.replace('```', '')
                response = response.split('<cluster table>')[-1].split('</cluster table>')[0].strip().split('\n')
                if 'markdown' in response[0]:
                    response = response[1:]
                if '---' in response[1]:
                    response = response[2:]
                else:
                    response = response[1:]
            except:
                continue
            min_rst = []
            min_rst_org_len = 0

            for r in response:
                try:
                    cur_label = r.split('|')[2].strip()
                except:
                    #print(" r.split('|')[2].strip()", r)
                    continue
                extract_di = extract_digits(r.split('|')[1].strip())
                if extract_di is not None:
                    r_idxs = list(range(extract_di[0], extract_di[1]))
                else:
                    r_idxs = r.split('|')[1].strip().split(',')

                new_texts = []
                for r_idx in r_idxs:
                    try:
                        ridx = int(r_idx)
                        new_texts.append(a_dict[ridx])
                    except:
                        continue

                        label_idx, label_name = get_label(a_dict[ridx], data)
                        rst.append({'text': a_dict[ridx], 'label_llm': cur_label, 'org_cluster': (lab_idx, li, ridx),
                                    'ground_label': (label_idx, label_name)})
                        min_rst.append(
                            {'text': a_dict[ridx], 'label_llm': cur_label, 'org_cluster': (lab_idx, li, ridx),
                             'ground_label': (label_idx, label_name)})
                        min_rst_org_len += 1
                        # for neighbor in text2data[a_dict[ridx]]['neighbor']:
                        #     min_rst.append(
                        #         {'text': neighbor['text'], 'label_llm': cur_label, 'org_cluster': (lab_idx, li, ridx),
                        #          'ground_label': (label_idx, label_name)})

                        #print(ridx, a_dict[ridx], (label_idx, label_name))

            if min_rst_org_len != len(lab):
                #print("not enough labels")
                labels_groups.append(lab[:int(len(lab) / 2)])
                labels_groups.append(lab[int(len(lab) / 2):])
                continue

            min_rsts.append(min_rst)

    return rst, min_rsts


if __name__ == '__main__':

    import os
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_path", default='./datasets/clinc/small_p.json', type=str,
                        help="the path of the dataset")
    parser.add_argument("--model_name", default='/root/autodl-tmp/LLM-Research/instructor_large', type=str)

    parser.add_argument("--mode_gen", default='small', type=str,
                        help='the mode that used for generation positive pairs')
    parser.add_argument('--save_llm_dir', default='./output/clinc',
                        help='the path of saving the data generated by llm')
    parser.add_argument('--use_llm', default='gpt3', type=str)
    args = parser.parse_args()

    print(f'start LPPG on {args.data_path}')
    data_path = args.data_path
    if not os.path.exists(args.save_llm_dir):
        os.makedirs(args.save_llm_dir)
    import json

    path_part = data_path.split('/')
    file_path = './prompt/prompts.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        instruction_prompt = json.load(file)
    instruction_ = instruction_prompt[path_part[-2]]

    from InstructorEmbedding import INSTRUCTOR

    model_name = args.model_name
    if torch.cuda.is_available():
        model_embed = INSTRUCTOR(model_name).cuda()
    else:
        model_embed = INSTRUCTOR(model_name)
    instruction_intent = instruction_

    model_embed.eval()

    n_data = read_jsonl(data_path)

    true_labels = set()
    for d in n_data:
        true_labels.add(d['label_text'])
    true_labels = list(true_labels)
    true_label_num = len(true_labels)

    ncluster = true_label_num
    true_labels = [item['label'] for item in n_data]
    test = [[instruction_intent, item['text']] for item in n_data]

    use_llm = args.use_llm

    llm_model = None
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if use_llm == 'llama3':
        llm_model = [AutoModelForCausalLM.from_pretrained(
            "/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct",
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        ), AutoTokenizer.from_pretrained("/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct")]

    mode_gen = args.mode_gen
    suffix = path_part[-2] + f'_{mode_gen}'
    #print(suffix)
    config_pre = args
    config_pre.all_data_v4 = True

    config_pre.nc = 20
    config_pre.seed = 42
    config_pre.n_classes = true_label_num

    #print(config_pre)

    according = 'according to the intent'
    target_text = 'intent'

    import pickle

    #print('seed', config_pre.seed)
    #print('use_llm', use_llm)
    prompt0 = """\
    Instruction
    ##Context
    - *Goal*  Your goal is to cluster the input utterances into meaningful categories **{according}**.
    - *Data*  The input data will be a markdown table with utterances including the following columns:
      - **id**   utterance index.
      - **utterance**  utterance·

    ##Requirements

    ### Format
    - Output clusters as a **markdown table**with each row as a category with the following columns:
      - **id**: all the utterance ids associated with this category
      - **description**: the {target_text} of the category that should be less than **4** words
    Here is an example of your output：
    ```markdown
    |id|description|
    |utterance ids|the {target_text} of the category|
    ```

    ###Quality
    - **No** **overlap** or **inclusion** among the categories.
    - **Do not include vague categories** such as "Other","General","Unclear","Miscellaneous" or "Undefined" in the cluster table.
    - Provide your answers between the tags: <cluster table>your generated cluster table</cluster table>, <explanation>explanation of your reasoning process within {n_exp} words</explanation>.
    - If the data points convey the **same** {target_text}, you should output just one category.
    - **Description** can **accurately** and **consistently** classify the Data **without ambiguity**. A data point must have and only belong to one category.

    # Data
    {data_table}

    # Output
    """

    #print(according, target_text, prompt0)
    if mode_gen == 'small':

        min_rsts_scatter, min_rsts = llm_label_v4(config_pre, n_data, model_embed,
                                                  according, target_text=target_text, prompt1=prompt0, llm=use_llm,
                                                  llm_model=llm_model)

        with open(f"{args.save_llm_dir}/min_rsts_{suffix}.pkl",
                  "wb") as file:
            pickle.dump(min_rsts, file)
        labels_all_set = set()
        for d in min_rsts_scatter:
            labels_all_set.add(d['label_llm'])
        labels_all = list(labels_all_set)
        with open(f"{args.save_llm_dir}/new2_{suffix}.pkl",
                  "wb") as file:
            pickle.dump(labels_all, file)

    if mode_gen == 'large':

        min_rsts_scatter, min_rsts = llm_label_v4_mini(config_pre, n_data, model_embed,
                                                       according, target_text=target_text, prompt1=prompt0, mini_avg=2,
                                                       llm=use_llm, llm_model=llm_model)

        with open(f"{args.save_llm_dir}/min_rsts_{suffix}.pkl",
                  "wb") as file:
            pickle.dump(min_rsts, file)
        labels_all_set = set()
        for d in min_rsts_scatter:
            labels_all_set.add(d['label_llm'])
        labels_all = list(labels_all_set)
        with open(f"{args.save_llm_dir}/new2_{suffix}.pkl",
                  "wb") as file:
            pickle.dump(labels_all, file)

    if mode_gen == 'random':

        min_rsts_scatter, min_rsts = llm_label_v4_random(config_pre, n_data, model_embed,
                                                         according, target_text=target_text, prompt1=prompt0,
                                                         llm=use_llm, llm_model=llm_model)

        with open(f"{args.save_llm_dir}/min_rsts_{suffix}.pkl",
                  "wb") as file:
            pickle.dump(min_rsts, file)
        labels_all_set = set()
        for d in min_rsts_scatter:
            labels_all_set.add(d['label_llm'])
        labels_all = list(labels_all_set)
        with open(f"{args.save_llm_dir}/new2_{suffix}.pkl",
                  "wb") as file:
            pickle.dump(labels_all, file)
    if mode_gen == 'random_all':

        min_rsts_scatter, min_rsts = llm_label_v4_random_all(config_pre, n_data, model_embed,
                                                             according, target_text=target_text, prompt1=prompt0,
                                                             llm=use_llm, llm_model=llm_model)

        with open(f"{args.save_llm_dir}/min_rsts_{suffix}.pkl",
                  "wb") as file:
            pickle.dump(min_rsts, file)
        labels_all_set = set()
        for d in min_rsts_scatter:
            labels_all_set.add(d['label_llm'])
        labels_all = list(labels_all_set)
        with open(f"{args.save_llm_dir}/new2_{suffix}.pkl",
                  "wb") as file:
            pickle.dump(labels_all, file)
 
    
    print(f'sucessfully generated positive pairs in {args.save_llm_dir}/min_rsts_{suffix}.pkl')
    print(f'sucessfully generated mini-cluster labels in {args.save_llm_dir}/new2_{suffix}.pkl')
