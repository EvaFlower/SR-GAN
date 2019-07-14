import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF 
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from ClusterSVDD.svdd_primal_sgd import SvddPrimalSGD
from ClusterSVDD.cluster_svdd import ClusterSvdd
from mixture_gaussian import data_generator
from mixture_gaussian import grid_data_generator


sample_size=10000
dg = data_generator()
enc_data = dg.sample(sample_size)
enc_data = np.transpose(enc_data, (1, 0))
print(enc_data.shape, type(enc_data))
svdds = []
n_cluster = 8
nu = 0.1
run = 1
#membership = np.random.randint(0, n_cluster, enc_data.shape[1])
#for c in range(n_cluster):
#    svdds.append(SvddPrimalSGD(nu))
#svdd = ClusterSvdd(svdds, nu=nu)
#cinds = svdd.fit(enc_data, init_membership=membership, max_svdd_iter=10000, max_iter=40)
#print(cinds)
file_name = 'csvdd_model_std1.sav'
#pickle.dump(svdd, open(file_name, 'wb'))
svdd = pickle.load(open(file_name, 'rb'))
svdds_c = []
svdds_radius2 = []
fig, ax = plt.subplots()
for i in range(svdd.clusters):
    c = svdd.svdds[i].c.reshape(1, -1)
    svdds_c.append(c)
    r = svdd.svdds[i].radius2
    svdds_radius2.append(r)
    color = plt.cm.Set3(i)
    print(color)
    c = c.reshape(-1)
    circle = plt.Circle(c, r, color=color)
    ax.add_artist(circle)
svdds_c = np.concatenate(svdds_c, axis=0).astype(np.float32)
#svdds_c = torch.tensor(svdds_c, device='cuda')
svdds_radius2 = np.array(svdds_radius2).astype(np.float32)
#svdds_radius2 = torch.tensor(svdds_radius2, device='cuda')
enc_data = np.transpose(enc_data, (1, 0))
print(svdds_radius2)
plt.scatter(enc_data[:, 0], enc_data[:, 1], c='b')
#plt.scatter(svdds_c[:, 0], svdds_c[:, 1], c='g')
fig.savefig('csvdd_r_std1.png')
plt.close()
#cinds = []
#all_scores = []
#for d, _ in data_loader:
#    images = Variable(d).cuda()
#    _, outputs = ae_net(images)
#    scores = []
#    for i in range(svdd.clusters):
#        s = (torch.sum((outputs-svdds_c[i])**2, dim=1)-svdds_radius2[i]).unsqueeze(1)
#        scores.append(s)
#    scores = torch.cat(scores, dim=1)
#    all_scores.append(scores)
#    cind = torch.min(scores, dim=1, keepdim=True)[0]
#    scores[scores>cind] = 0
#    print(torch.sum(scores, dim=0)/64)
#    cinds.append(cind)
#all_scores = torch.cat(all_scores, dim=0)
#cinds = torch.cat(cinds, dim=0)
#cinds = cinds.repeat(1, all_scores.size(1))
#print(cinds.size(), all_scores.size())
#all_scores[all_scores>cinds] = 0
#print(all_scores[:20])
##_, cinds = svdd.predict(enc_data)
#print(torch.mean(all_scores, dim=0))
