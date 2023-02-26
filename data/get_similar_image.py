import json
import time
import torch
from tqdm import tqdm

start =time.time()
with open('./image_feature/resnet18_features.json',"r", encoding="utf-8") as f:
    features = json.load(f)
end = time.time()
print('time cost of reading image features: {}'.format(end-start))
print(type(features))  # dict  image name: feature(list)

# load the image name from user
with open('image_name_user_provide.json',"r", encoding="utf-8") as f:
    image_from_user = json.load(f)
image_feature_user = []
for img in image_from_user:
    f = features[img+'.jpg']  # list  # len 512
    image_feature_user.append(features[img+'.jpg'])


# load the image name of db
with open('image_map.json',"r", encoding="utf-8") as f:
    image_from_db_map = json.load(f)
image_from_db = list(image_from_db_map.keys())

image_feature_db = []
for img in image_from_db:
    image_feature_db.append(features[img+'.jpg'])


def get_sim(target, features):
    sim = torch.zeros(target.size(0), features.size(0))
    print(sim.size())

    for i in tqdm(range(target.size(0))):
        score = torch.cosine_similarity(target[i].unsqueeze(0), features)  # 计算每一个元素与给定元素的余弦相似度
        sim[i] = score
    return sim

image_feature_user = torch.Tensor(image_feature_user) # torch.Size([793, 512])
image_feature_db = torch.Tensor(image_feature_db) # torch.Size([85527, 512])

start = time.time()
sim = get_sim(image_feature_user, image_feature_db)
print(sim.size()) # torch.Size([793, 85527])
end = time.time()
print('time cost of geting similar images: {}'.format(end-start))
max_sim = torch.max(sim,1)[0]
# min: 0.7253, max: 0.9367
def save_json(data, save_name):
    with open(save_name+'.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(data, indent=4))


sort = torch.argsort(sim, dim=1)
max_idx = sort[:,0]
# the max image id in db
image_user_sim_db_id = {}
for thr in [0.75, 0.8, 0.85, 0.9]:
    for img, maxid, sim in zip(image_from_user, max_idx, max_sim):
        if sim>=thr:
            image_name = image_from_db[maxid]
            image_user_sim_db_id[img] = image_from_db_map[image_name]['className']
    save_json(image_user_sim_db_id, 'image_user_sim_db_id'+'_'+str(thr))
