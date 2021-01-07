import os
import numpy as np
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset

np.random.seed(7)


class Loader(Dataset):
    def __init__(self, root='../generated/mini/gen_None_None/real'):
        self.root = root
        files = np.asarray(os.listdir(self.root)).astype(np.int32)
        files.sort()
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open('{}/{}'.format(self.root, str(self.files[idx])), 'rb') as file:
            loaded = pickle.load(file)
        return loaded


split_ratio = 0.1
name1, type1 = 'gen_None_None', 'train'
name2, type2 = 'gen_30.0_None', 'fake'
dataset1 = Loader(root='../generated/v1.0-trainval/{}'.format(os.path.join(name1, type1)))
dataset2 = Loader(root='../generated/v1.0-trainval/{}'.format(os.path.join(name2, type2)))

save_dir = '../generated/mixed/{}_{}_{}_{}_ratio_{}'.format(name1, type1, name2, type2, split_ratio)
if os.path.isdir(save_dir):
    print('directory exist')
    exit(0)
else:
    os.mkdir(save_dir)

# count agents
agent1_num = 0
agent2_num = 0
for data in dataset1:
    agent1_num += np.sum(data[4])
for data in dataset2:
    agent2_num += np.sum(data[4])

print('path 1: {}'.format(dataset1.root))
print('path 2: {}'.format(dataset2.root))
print('agents1: {}, agents2: {}'.format(agent1_num, agent2_num))

target_agent1_num = min(agent1_num, int(agent2_num * split_ratio / (1-split_ratio)))
target_agent2_num = min(agent2_num, int(agent1_num * (1-split_ratio) / split_ratio))

# save
idx = 0

num_temp = 0
for data in tqdm(dataset1, total=len(dataset1), desc='saving first'):
    with open('{}/{}'.format(save_dir, idx), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    idx += 1
    num_temp += np.sum(data[4])
    if num_temp >= target_agent1_num:
        break

num_temp = 0
for data in tqdm(dataset2, total=len(dataset2), desc='saving second'):
    with open('{}/{}'.format(save_dir, idx), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    idx += 1
    num_temp += np.sum(data[4])
    if num_temp >= target_agent2_num:
        break
