import numpy as np
import warnings
import os
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def load_off(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        if 'OFF' in first_line:
            if first_line == 'OFF':
                n_verts, n_faces, n_edges = tuple(map(int, file.readline().strip().split(' ')))
            else:
                parts = first_line[3:].strip().split(' ')
                n_verts, n_faces, n_edges = tuple(map(int, parts))
        else:
            raise ValueError('Not a valid OFF header')

        vertices = []
        for i in range(n_verts):
            vertices.append(list(map(float, file.readline().strip().split(' '))))

        faces = []
        for i in range(n_faces):
            faces.append(list(map(int, file.readline().strip().split(' '))))

        return np.array(vertices), np.array(faces)


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.split = split
        self.categories = os.listdir(self.root)
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = []
        shape_ids['test'] = []

        for category in self.categories:
            train_files = os.listdir(os.path.join(self.root, category, 'train'))
            test_files = os.listdir(os.path.join(self.root, category, 'test'))
            shape_ids['train'] += [(category, os.path.join(self.root, category, 'train', file)) for file in train_files]
            shape_ids['test'] += [(category, os.path.join(self.root, category, 'test', file)) for file in test_files]

        assert (split == 'train' or split == 'test')
        self.datapath = shape_ids[split]
        print(f'The size of {split} data is {len(self.datapath)}')

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.categories.index(fn[0])
            cls = np.array([cls]).astype(np.int32)
            point_set, _ = load_off(fn[1])

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            if self.split == 'train':
                train_idx = np.array(range(point_set.shape[0]))
                point_set = point_set[train_idx[:self.npoints], :]
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)
