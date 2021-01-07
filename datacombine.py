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


name1, type1 = 'None_None', 'train'
name2, type2 = 'None_None', 'fake'
dataset1 = Loader(root='../generated/v1.0-trainval/{}'.format(os.path.join('gen_'+name1, type1)))
dataset2 = Loader(root='../generated/v1.0-trainval/{}'.format(os.path.join('gen_'+name2, type2)))

save_dir = '../generated/combined/{}_{}_{}_{}'.format(name1, type1, name2, type2)
if os.path.isdir(save_dir):
    print('directory exist')
    exit(0)
else:
    os.mkdir(save_dir)

print('path 1: {}'.format(dataset1.root))
print('path 2: {}'.format(dataset2.root))

# save
idx = 0
for data in tqdm(dataset1, total=len(dataset1), desc='saving first'):
    with open('{}/{}'.format(save_dir, idx), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    idx += 1
for data in tqdm(dataset2, total=len(dataset2), desc='saving second'):
    with open('{}/{}'.format(save_dir, idx), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    idx += 1
