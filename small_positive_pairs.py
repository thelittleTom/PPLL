from argparse import ArgumentParser
from InstructorEmbedding import INSTRUCTOR
import torch
import os
import pickle
import json
from sklearn.cluster import KMeans

parser = ArgumentParser()
parser.add_argument("--data_path", default='./datasets/clinc/small_p.json', type=str,
                    help="the path of the dataset")
parser.add_argument("--model_name", default='/root/autodl-tmp/LLM-Research/instructor_large', type=str)
parser.add_argument("--save_dir", default='./output/clinc')
parser.add_argument('--suffix', default='clinc_small', type=str)
args = parser.parse_args()

data_path = args.data_path
model_name = args.model_name
if torch.cuda.is_available():
    model_embed = INSTRUCTOR(model_name).cuda()
else:
    model_embed = INSTRUCTOR(model_name)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


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
data = read_jsonl(data_path)
nc = int(len(data) / 1.2)

batch_size = 2000
embeddings = []
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    sentences = [[instruction_, item['text']] for item in batch]
    sentence_vectors = model_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    embeddings.extend(sentence_vectors)



kmeans = KMeans(n_clusters=nc, init='k-means++', )
kmeans.fit(embeddings)
cluster_labels = kmeans.labels_

centroids = kmeans.cluster_centers_
prediction = {}
for d, l in zip(data, cluster_labels):
    if l not in prediction:
        prediction[l] = []
    prediction[l].append(d)
train_samples_set = set()
s, sn = 0, 0
for k, v in prediction.items():

    for i, vi in enumerate(v):
        for j in range(i, len(v)):
            if i != j:
                if vi['label'] == v[j]['label']:
                    s += 1
                else:
                    sn += 1

                train_samples_set.add(vi['text'] + '-|)$^' + v[j]['text'])
train_samples = [_.split("-|)$^") for _ in train_samples_set]
print('the accuracy of small positive pairs for auto evaluation is',s / (s + sn))
print('the size of positive pairs for auto evaluation',len(train_samples))

with open(f"{args.save_dir}/kmeans_pairs2_{args.suffix}.pkl",
          "wb") as file:
    pickle.dump(train_samples, file)
print(f'successfully saved the positive pairs for auto evaluation to {args.save_dir}/kmeans_pairs2_{args.suffix}.pkl')