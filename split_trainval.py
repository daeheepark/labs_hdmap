import os
import numpy as np
from tqdm import tqdm
import pickle
from torch.utils.data import  Dataset

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


split_ratio = 0.8
dataset = Loader(root='../generated/v1.0-trainval/gen_None_30.0/real')

save_dir = '../generated/v1.0-trainval/gen_None_30.0'
train_dir = os.path.join(save_dir, 'train')
val_dir = os.path.join(save_dir, 'val')

if os.path.isdir(train_dir) or os.path.isdir(val_dir):
    print('directory exist')
    exit(0)
else:
    os.mkdir(train_dir)
    os.mkdir(val_dir)

sz = len(dataset)
shuffled_idx = np.random.choice(sz, sz, replace=False)

train_num = int(sz*split_ratio)
val_num = sz - train_num

train_idx = shuffled_idx[:train_num]
val_idx = shuffled_idx[train_num:]

# save train dataset
for idx in tqdm(train_idx, total=train_num, desc='save train'):
    with open('{}/{}'.format(train_dir, idx), 'wb') as f:
        pickle.dump(dataset[idx], f, pickle.HIGHEST_PROTOCOL)

# save valid dataset
for idx in tqdm(val_idx, total=val_num, desc='save val'):
    with open('{}/{}'.format(val_dir, idx), 'wb') as f:
        pickle.dump(dataset[idx], f, pickle.HIGHEST_PROTOCOL)

