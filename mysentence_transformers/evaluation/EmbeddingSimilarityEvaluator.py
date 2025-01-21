import numpy

from . import SentenceEvaluator, SimilarityFunction
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
from ..readers import InputExample
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering
def calculate_acc_cls(true_labels, cluster_labels):
    import numpy as np
    from sklearn.metrics import accuracy_score
    classes = np.unique(true_labels)
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    accs = {}
    # 计算每个类别的准确率
    for cls in classes:
        # 找出属于该类别的所有索引
        cls_indices = np.where(true_labels == cls)[0]
        true_cls = true_labels[cls_indices]
        pred_cls = cluster_labels[cls_indices]

        accuracy = accuracy_score(true_cls, pred_cls)

        # print(f"Accuracy for class {cls}  : {accuracy:.4f}")
        accs[int(cls)]=[accuracy]
    return accs
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
logger = logging.getLogger(__name__)

class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self,
                 show_progress_bar: bool = False,
                 data_path:str='',instruction_:str='',
                 eval_kmeans_samples=None,mini_cluster_labels=None):




        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.data_path=data_path
        self.instruction_=instruction_

        self.eval_kmeans_samples= eval_kmeans_samples

        self.mini_cluster_labels=mini_cluster_labels


    def eval(self,model_embed , eval_add_label=True):
        import jsonlines

        raw_text='input'
        def read_jsonl(path):
            content = []
            with jsonlines.open(path, "r") as json_file:
                for obj in json_file.iter(type=dict, skip_invalid=True):
                    content.append(obj)
            return content


        instruction_intent = self.instruction_
        model_embed.eval()
        data_path =self.data_path
        n_data = read_jsonl(data_path)
        text2id = {}
        for i, d in enumerate(n_data):
            text2id[d['text']] = i

        true_labels = [item['label'] for item in n_data]

        import pickle
        if  eval_add_label  :
            with open(self.mini_cluster_labels,
                      "rb") as file:
                # 使用 pickle.load() 恢复嵌套列表
                labels_v4 = pickle.load(file)
            fake_n_data = []
            for l in labels_v4:
                fake_n_data.append({'text': l})
            n_data = n_data + fake_n_data

        test = [[instruction_intent, item['text']] for item in n_data]
        # list_data_embedding = model_embed.encode(test)
        batch_size = 2000
        list_data_embedding = []

        # 将数据按batch_size分块
        for i in  range(0, len(n_data), batch_size) :
            batch = n_data[i:i + batch_size]

            sentences = [[instruction_intent, item['text']] for item in batch]

            sentence_vectors = model_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
            # 存储句子向量和原始数据
            list_data_embedding.extend(sentence_vectors)


        all_acc = []
        all_acc_cls=None
        all_NMI = []
        all_scc=[]
        all_score=[]
        all_sss=[]
        n_clusters=len(set(true_labels))

        from sklearn.cluster import KMeans
        print(f'start kmeans {n_clusters}')
        for seed in  [100,13, 21, 36, 42]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, )

            # kmeans = KMeans(n_clusters=77, max_iter=10000, random_state=42, init=init_c,, init='k-means++', )
            kmeans.fit(list_data_embedding)
            # 获取聚类结果
            cluster_labels = kmeans.labels_[:len(true_labels)]
            acc, new_cluster_labels = calculate_acc(true_labels, cluster_labels)
            acc_cls=calculate_acc_cls(true_labels,new_cluster_labels)
            if all_acc_cls is None:
                all_acc_cls=acc_cls
            else:
                for k,v in acc_cls.items():
                    all_acc_cls[k].extend(v)

            # acc  = clustering_accuracy_score(true_labels, cluster_labels)
            all_acc.append(acc)
            score,score_no=0,0
            for ts in self.eval_kmeans_samples:
                a_idx = text2id[ts[0]]
                b_idx = text2id[ts[1]]
                if new_cluster_labels[a_idx] == new_cluster_labels[b_idx]:
                    score += 1
                else:
                    score_no += 1
            if score+score_no==0:
                score=1
            all_score.append(score/(score_no+score))
            print(f'kmean   accuracy:{acc}')

            print('score:',score/(score_no+score))
            overall_score = silhouette_score(list_data_embedding[:len(true_labels)], cluster_labels)
            print("Overall Silhouette Score:", overall_score)
            all_scc.append(overall_score)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
            print("NMI Score:", nmi_score)
            all_NMI.append(nmi_score)
            all_sss.append((score / (score_no + score)) * (1 + overall_score))
        print('--average acc',np.mean(all_acc))
        print('--average ss',np.mean(all_scc))
        print('--average nmi',np.mean(all_NMI))
        print('--average score', np.mean(all_score))
        print('--average sss',np.mean(all_sss))
        for k,v in all_acc_cls.items():
            print(f'{k} : {np.mean(v)}')
        return np.mean(all_sss),np.mean(all_score)



    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1)  :
        if epoch != -1:

            out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluating the model" + out_txt)



        print('test-data')
        sss,score=self.eval(model,  eval_add_label=False )
        print('lv4')
        sss2,score2=self.eval(model, eval_add_label=True )
        if sss>=sss2:
            return {'sss':sss,'score':score,'mini_cluster_labels':False}
        else:
            return {'sss':sss2,'score':score2,'mini_cluster_labels':True}
