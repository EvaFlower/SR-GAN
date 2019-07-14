import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import ipdb
import itertools


"""
    generate 2d gaussian around a circle
"""
class data_generator(object):
    def __init__(self):

        n = 8
        radius = 2.0
        std = 0.02 #0.02
        delta_theta = 2*np.pi / n
         
        centers_x = []
        centers_y = []
        for i in range(n):
            centers_x.append(radius*np.cos(i*delta_theta))
            centers_y.append(radius*np.sin(i*delta_theta))

        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)

        p = [1./n for _ in range(n)]

        self.p = p
        self.size = 2
        self.n = n
        self.std = std
        self.centers = np.concatenate([centers_x, centers_y], 1)

    # switch to random distribution (harder)
    def random_distribution(self, p=None):
        if p is None:
            p = [np.random.uniform() for _ in range(self.n)]
            p = p / np.sum(p)
        self.p = p

    # switch to uniform distribution
    def uniform_distribution(self):
        p = [1./self.n for _ in range(self.n)]
        self.p = p

    def sample(self, N):
        n = self.n
        std = self.std
        centers = self.centers

        ith_center = np.random.choice(n, N,p=self.p)
        sample_centers = centers[ith_center, :]
        sample_points = np.random.normal(loc=sample_centers, scale=std)
        return sample_points.astype('float32')
        
    def sample_certain_gauss(self, N, i):
        std = self.std
        centers = self.centers
        ith_center = np.ones(N, dtype=int)*i
        sample_centers = centers[ith_center, :]
        sample_points = np.random.normal(loc=sample_centers, scale=std)
        return sample_points.astype('float32')

    def sample_with_label(self, N):
        samples = []
        labels = []
        for i in range(self.n):
            n = int(N/self.n)
            samples.append(self.sample_certain_gauss(n, i))
            labels.append(np.ones(n, dtype=int)*i)
        samples = np.concatenate(samples, axis=0)
        labels = np.concatenate(labels, axis=0)
        rands = np.random.permutation(range(labels.shape[0]))
        return samples[rands, :], labels[rands]
       

class grid_data_generator(object):
    def __init__(self):

        n = 100
        space = 0.2  #2
        std = 0.01 #0.02  #0.05
        p = [1./n for _ in range(n)]

        self.p = p
        self.size = 2
        self.n = n
        self.std = std
        grid_range = int(np.sqrt(n))
        modes = np.array([np.array([i, j]) for i, j in
                          itertools.product(range(-grid_range + 1, grid_range, 2),
                                            range(-grid_range + 1, grid_range, 2))],
                         dtype=np.float32)
        modes = modes * space / 2.
        self.centers = modes

    # switch to random distribution (harder)
    def random_distribution(self, p=None):
        if p is None:
            p = [np.random.uniform() for _ in range(self.n)]
            p = p / np.sum(p)
        self.p = p

    # switch to uniform distribution
    def uniform_distribution(self):
        p = [1./self.n for _ in range(self.n)]
        self.p = p

    def sample(self, N):
        n = self.n
        std = self.std
        centers = self.centers

        ith_center = np.random.choice(n, N,p=self.p)
        sample_centers = centers[ith_center, :]
        sample_points = np.random.normal(loc=sample_centers, scale=std)
        return sample_points.astype('float32')

    def sample_certain_gauss(self, N, i):
        std = self.std
        centers = self.centers
        ith_center = np.ones(N, dtype=int)*i
        sample_centers = centers[ith_center, :]
        sample_points = np.random.normal(loc=sample_centers, scale=std)
        return sample_points.astype('float32')

    def sample_with_label(self, N):
        samples = []
        labels = []
        for i in range(self.n):
            n = int(N/self.n)
            samples.append(self.sample_certain_gauss(n, i))
            labels.append(np.ones(n)*i)
        samples = np.concatenate(samples, axis=0)
        labels = np.concatenate(labels, axis=0)
        rands = np.random.permutation(range(labels.shape[0]))
        return samples[rands, :], labels[rands]


def plot(points):
    plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1], c='g', edgecolor='none')
    plt.axis('off')
    plt.savefig('grid_samples.png')
    plt.close()

def main():
    gen = grid_data_generator()
    sample_points = gen.sample_certain_gauss(1000, 0)
    sample_points_2 = gen.sample_certain_gauss(1000, 7)
    sample_points_3 = gen.sample(1000)
    sample_points = np.concatenate((sample_points, sample_points_2, sample_points_3), axis=0)
    plot(sample_points)

if __name__ == '__main__':
    main()
