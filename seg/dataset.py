import numpy as np
import os
from torch.utils.data import Dataset


class S3dis(Dataset):
    def __init__(self, root, split, loop, npoints=24000, voxel_size=0.04, test_area=5, transforms=None):
        super(S3dis, self).__init__()
        self.root = root
        self.split = split
        self.loop = loop
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.transforms = transforms
        self.idx_to_class = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 
                5: 'window', 6: 'door', 7: 'table', 8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'}
        
        room_list = os.listdir(root)
        if split == 'train':
            self.room_list = list(filter(lambda x : f'Area_{test_area}' not in x, room_list))
        else:
            self.room_list = list(filter(lambda x : f'Area_{test_area}' in x, room_list))
    
    def __len__(self):
        return len(self.room_list) * self.loop

    def fnv_hash_vec(self, arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * \
            np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    def voxel_grid_sampling(self, pos):
        """
        pos.shape = (n, 3)
        """
        voxel_indices = np.floor(pos / self.voxel_size)
        
        voxel_hash = self.fnv_hash_vec(voxel_indices)
        sort_idx = voxel_hash.argsort()
        hash_sort = voxel_hash[sort_idx]
        
        _, counts = np.unique(hash_sort, return_counts=True)
        if self.split == 'test':   # test时需要的东西和train，val时不同
            return sort_idx, counts
        
        idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1]) + np.random.randint(0, counts.max(), counts.size) % counts
        return sort_idx[idx_select]
    
    def __getitem__(self, index):
        room = os.path.join(self.root, self.room_list[index % len(self.room_list)])
        points = np.load(room)
        
        # 大家都这样做
        points[:, 0:3] = points[:, 0:3] - np.min(points[:, 0:3], axis=0)
        
        if self.split == 'test':
            sort_idx, counts = self.voxel_grid_sampling(points[:, 0:3])
            pos, color, y = points[:, 0:3], points[:, 3:-1], points[:, -1]
            pos, color, y = pos.astype(np.float32), color.astype(np.float32), y.astype(np.int64)
            return pos, color, y, sort_idx, counts, room
        
        # train, val的流程
        sample_indices = self.voxel_grid_sampling(points[:, 0:3])
        pos, color, y = points[sample_indices, 0:3], points[sample_indices, 3:-1], points[sample_indices, -1]
        
        # 是否指定了npoints
        if self.npoints:
            n = len(sample_indices)
            if n > self.npoints:
                init_idx = np.random.randint(n)
                crop_indices = np.argsort(np.sum(np.square(pos - pos[init_idx]), 1))[:self.npoints]
            elif n < self.npoints:
                temp = np.arange(n)
                pad_choice = np.random.choice(n, self.npoints - n)
                crop_indices = np.hstack([temp, temp[pad_choice]])
            else:
                crop_indices = np.arange(n)
            
            # 打乱
            np.random.shuffle(crop_indices)
        
            pos, color, y = pos[crop_indices], color[crop_indices], y[crop_indices]
        
        pos = pos - pos.min(0)
        if self.transforms:
            pos, color, _ = self.transforms(pos, color, None)
        
        pos, color, y = pos.astype(np.float32), color.astype(np.float32), y.astype(np.int64)
        return pos, color, y
