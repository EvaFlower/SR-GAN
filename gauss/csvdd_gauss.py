import sys
sys.path.append('..')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import sklearn.metrics as metrics
import numpy as np
import pickle

from ClusterSVDD.svdd_dual_qp import SvddDualQP
from ClusterSVDD.svdd_primal_sgd import SvddPrimalSGD
from ClusterSVDD.cluster_svdd import ClusterSvdd

from mixture_gaussian import data_generator
from mixture_gaussian import grid_data_generator


def load_data_set(fname, num_data, outlier_frac, train_inds):
    from sklearn.datasets import load_svmlight_file
    X, y = load_svmlight_file(fname)

    print(X.shape)
    y -= np.min(y) # classes should start from zero: 0,1,2,3,...
    inds = np.array([], dtype='i')
    for i in range(int(max(y))+1):
        inds = np.append(inds, np.where(y == i)[0])

    print(inds.shape)
    X = X.toarray()
    X = X[inds, :]
    y = y[inds]

    inds = np.random.permutation(range(y.size))
    X = X[inds[:num_data], :].T
    y = y[inds[:num_data]]

    # induce anomalies
    anoms = int(float(num_data)*outlier_frac)
    X[:, :anoms] = 1.*(np.random.rand(X.shape[0], anoms)*2.-1.)
    y[:anoms] = -1

    print(np.unique(y))
    return X, y


if __name__ == '__main__':
    nu = 0.5
    k = 50
    outlier_frac = 0.02
    num_train = 16000
    num_test = 5000
    num_val = 5000
    use_kernels = False

    train = np.array(range(num_train), dtype='i')
    val = np.array(range(num_train, num_train+num_val), dtype='i')
    test = np.array(range(num_train+num_val, num_train + num_val + num_test), dtype='i')
    
    dg = grid_data_generator()
    inds = np.random.permutation(range(num_test + num_train + num_val))
    data, y = dg.sample_with_label(int(num_test+num_train+num_val))
    data = data.T
    membership = np.random.randint(0, k, y.size)
    svdds = list()
    for l in range(k):
        if use_kernels:
            svdds.append(SvddDualQP('rbf', 20.0, nu))
        else:
            svdds.append(SvddPrimalSGD(nu))
    svdd = ClusterSvdd(svdds)
    svdd.fit(data[:, train].copy(), max_iter=100, max_svdd_iter=100000, init_membership=membership[train])
    file_name = 'csvdd_model_grid_c'+str(k)+'.sav'
    pickle.dump(svdd, open(file_name, 'wb'))
    svdd = pickle.load(open(file_name, 'rb'))
    # test error
    print(data.shape, test[-1])
    scores, classes = svdd.predict(data[:, test].copy())
    
    ari = metrics.cluster.adjusted_rand_score(y[test], classes)
    if nu < 1.0:
        inds = np.where(scores <= 0.)[0]

        ari = metrics.cluster.adjusted_rand_score(y[test[inds]], classes[inds])
    ari = ari

    # ...and anomaly detection accuracy
    fpr, tpr, _ = metrics.roc_curve(np.array(y[test]<0., dtype='i'), scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # validation error
    scores, classes = svdd.predict(data[:, val].copy())
    # evaluate clustering abilities
    # inds = np.where((y[val] >= 0))[0]
    # val_aris[n, i, k] = metrics.cluster.adjusted_rand_score(y[val[inds]], classes[inds])

    ari = metrics.cluster.adjusted_rand_score(y[val], classes)
    if nu < 1.0:
        inds = np.where(scores <= 0.)[0]
        ari = metrics.cluster.adjusted_rand_score(y[val[inds]], classes[inds])
    val_ari = ari

    # ...and anomaly detection accuracy
    fpr, tpr, _ = metrics.roc_curve(np.array(y[val]<0., dtype='i'), scores, pos_label=1)
    val_auc = metrics.auc(fpr, tpr)
    print('test: ', ari, auc)
    print('val: ', val_ari, val_auc)
    svdds_c = []
    for i in range(svdd.clusters):
        c = svdd.svdds[i].c.reshape(1, -1)
        svdds_c.append(c)
    svdds_c = np.concatenate(svdds_c, axis=0).astype(np.float32)
    plt.scatter(data[:, train].T[:, 0], data[:, train].T[:, 1], c='g')
    plt.scatter(svdds_c[:, 0], svdds_c[:, 1], c='r')
    plt.savefig('grid_50_csvdd.png')
