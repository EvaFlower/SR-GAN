import os
import time
import copy
import torch
import datetime
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch import autograd
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
import pickle
from datetime import datetime as dt
import collections

#from msssim import *
from model import MLP_D_
from model import MLP_G_
from mixture_gaussian import data_generator
from mixture_gaussian import grid_data_generator

import tensorflow as tf

from ClusterSVDD.svdd_primal_sgd import SvddPrimalSGD
from ClusterSVDD.cluster_svdd import ClusterSvdd


class Solver(object):
    
    def __init__(self, data_loader, config):
        
        # Data loader
        self.data_loader = data_loader
        
        # Model hyper-parameters
        self.dataset = config.dataset
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_f_dim = config.g_f_dim
        self.d_f_dim = config.d_f_dim
        self.image_size = config.image_size
        self.image_channel = config.image_channel
        self.lambda_gp = config.lambda_gp
        self.num_gen = config.num_gen
        self.num_dis = config.num_dis
        self.num_ocsvm = config.num_ocsvm
        self.ms_num_image = config.ms_num_image
        self.lambda_ncl = config.lambda_ncl
        self.lambda_s = config.lambda_s
        self.softmax_beta = config.softmax_beta
        self.n_viz = config.n_viz
        self.mog_scale = config.mog_scale
        self.num_class = config.num_class
        self.num_cluster = config.num_cluster
   
        # Training settings
        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.d_iters = int(config.d_iters)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.gum_t = config.gum_temp
        self.gum_orig = config.gum_orig
        self.gum_t_decay = config.gum_temp_decay
        self.step_t_decay = config.step_t_decay
        self.start_anneal = config.start_anneal
        self.min_t = config.min_temp
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model
        self.ngpu = config.ngpu
        self.run = config.run

        # Test settings
        self.test_size = config.test_size
        self.test_model = config.test_model
        self.test_ver = config.test_ver
        self.version = config.version
        self.result_path = os.path.join(config.result_path, self.version)
        self.nrow = config.nrow
        self.ncol = config.ncol
        
        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.score_epoch = config.score_epoch
        self.score_start = config.score_start

        #load-balancing
        self.load_balance = config.load_balance
        self.balance_weight = config.balance_weight
        self.matching_weight = config.matching_weight
        self.lambda_s = config.lambda_s
        self.lambda_d = config.lambda_d
        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.model_save_path_test = config.model_save_path

        
        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()


    def build_model(self):
        
        # Define summarywriter
        if self.use_tensorboard == True:
            now = dt.now()
            self.log_path = os.path.join(self.log_path, now.strftime(" \
                %Y%m%d-%H%M%S"))
            self.writer = SummaryWriter(self.log_path)
        # Define the generator and discriminator

        #Generator will output batch x num_gen x 3 x imsize x imsize
        #self.G = Generator(self.image_size, self.image_channel, self.z_dim, self.g_conv_dim).cuda()
        #self.D = Discriminator(self.image_size, self.image_channel, self.num_dis, self.d_conv_dim).cuda()
        file_name = 'csvdd_'+str(self.num_cluster)+'c_nu05_'+self.dataset+'_'+str(self.run)+'.sav'
        svdd = pickle.load(open(file_name, 'rb'))
        svdds_c = []
        svdds_radius2 = []
        for i in range(svdd.clusters):
            c = svdd.svdds[i].c.reshape(1, -1)
            svdds_c.append(c)
            r = svdd.svdds[i].radius2
            svdds_radius2.append(r)
        svdds_c = np.concatenate(svdds_c, axis=0).astype(np.float32)
        svdds_c = torch.tensor(svdds_c, device='cuda')
        svdds_radius2 = np.array(svdds_radius2).astype(np.float32)
        svdds_radius2 = torch.tensor(svdds_radius2, device='cuda')
        self.svdds_c = svdds_c
        self.svdds_radius2 = svdds_radius2
        self.svdd_clusters = svdd.clusters
        self.G = MLP_G_(self.z_dim, self.g_f_dim, self.ngpu).cuda()
        self.D = MLP_D_(self.d_f_dim, self.ngpu).cuda()
        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        # print networks
        #print(self.G)
        #print(self.D)
        #print(self.gate)

    def many_to_one(self, outputs, gumbel_out, num_dis):
        # if num_dis == 1:
        #     gumbel_out = Variable(torch.ones(z.size(0), 1, 3, self.image_size, self.image_size)).cuda()
        #print(outputs.size())
        #print(gumbel_out.size())
        if num_dis == 1:
            output = outputs
        else:
            output  = torch.sum(torch.mul(outputs, gumbel_out), dim=1)  # batch x 3 x imsize x imsize
        return output

    def load_pretrained_model(self):
        
        checkpoint = torch.load(os.path.join(self.model_save_path, '{}.pth'.format(self.pretrained_model)))
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        #self.gate.load_state_dict(checkpoint['gate'])
        #self.gum_t = check_point['gum_t']
        
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def numpy2var(self, x, grad=False):
        x = torch.FloatTensor(x)
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=grad)
        
    def tensor2var(self, x, grad=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=grad)
    
    def var2tensor(self, x):
        return x.data.cpu()

    def var2numpy(self, x):
        return x.data.cpu().numpy()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        
    def init_gamma(self):
        gamma = torch.empty(self.batch_size, self.num_ocsvm)
        nn.init.normal_(gamma)
        gamma = F.softmax(gamma, dim=1)
        return gamma
    
    def loss_function(self, samples, clusters):
        cluster = Variable(torch.from_numpy(clusters[0, :])).cuda()
        cluster = cluster.repeat(samples.size(0), 1)
        to_clusters = torch.norm((samples-cluster), 2, 1) \
            .unsqueeze(1)
        for i in range(1, clusters.shape[0]):
            cluster = Variable(torch.from_numpy(clusters[i, :])).cuda()
            cluster = cluster.repeat(samples.size(0), 1)
            temp = torch.norm((samples-cluster), 2, 1).unsqueeze(1)
            to_clusters = torch.cat([to_clusters, temp], dim=1)
        return to_clusters.squeeze()
   
    def train_outlier(self):
        dg = data_generator()
        samples = dg.sample(1000)
        
        for step in range(25000):
            samples = dg.sample(512)
            samples = self.tensor2var(torch.FloatTensor(samples))
            weights, _, _ = self.gate(samples, self.gum_t, True)
            losses =  self.outlier(samples)
            #losses = torch.clamp(1-losses, min=0)
            loss = torch.mean(torch.mul(losses**2, weights))
            weight_loss = 0
            #for i in range(len(self.outlier.all_layers)):
                #weight_loss += torch.mean(self.outlier.all_layers[i] \
                    #[0].weight ** 2)
            loss += 0.01 * weight_loss
            self.reset_grad()
            loss.backward()
            self.gate_optimizer.step()
            self.outlier_optimizer.step()
            
        xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5,\
            5, 500))
        zz_to_clusters = self.outlier(self.tensor2var(torch. \
            FloatTensor(np.c_[xx.ravel(), yy.ravel()])))
        zz_weights, _, _ = self.gate(self.tensor2var(torch.FloatTensor \
            (np.c_[xx.ravel(), yy.ravel()])), self.gum_t, True)

        zz = self.var2numpy(torch.sum(torch.mul(zz_to_clusters, \
            zz_weights),dim=1))
        zz = zz.reshape(xx.shape)
        plt.figure(1)
        plt.scatter(samples[:, 0], samples[:, 1], color='b')
        ct = plt.contour(xx, yy, zz, linewidths=2)
        plt.clabel(ct, inline=1, fontsize='smaller')
        plt.xlim((xx.min(), xx.max()))
        plt.ylim((yy.min(), yy.max()))
        plt.savefig(os.path.join(self.result_path, '{}_linear_outlier_2.png'. \
            format(1)))
    
    def calculate_csvdd_support(self, inputs):
        outputs = inputs
        scores = []
        for i in range(self.svdd_clusters):
            s = (torch.sum((outputs-self.svdds_c[i])**2, dim=1)-self.svdds_radius2[i]).unsqueeze(1)
            scores.append(s)
        scores = torch.cat(scores, dim=1)
        beta = self.softmax_beta
        #support = torch.mul(F.softmax(-scores*beta, dim=1), scores)
        s_max = torch.max(-scores, dim=1, keepdim=True)[0].detach()
        support = F.softmax(-scores*beta, dim=1)
        #print(support.data[0])
        support = torch.mean(support, dim=0)
        #mins = torch.min(scores, dim=1, keepdim=True)[0]
        #cinds = torch.eq(scores, mins).type_as(scores).detach()
        #support = torch.mul(scores, cinds)
        #cinds = cinds.repeat(1, scores.size(1))
        #support = (cinds-scores).detach()+scores
        #support = torch.mean(support, dim=0)
        return support

    def test_quality(self):
        if self.dataset == 'grid':
            dg = grid_data_generator()
        elif self.dataset == 'ring':
            dg = data_generator()
        #z = self.tensor2var(torch.randn(self.n_viz, 
        #    self.z_dim))
        #fake_samples = self.var2numpy(self.G(z))
        #fake_samples = self.var2numpy(fake_samples)
        #samples = dg.sample(self.n_viz)
        z = self.tensor2var(torch.randn(self.n_viz, self.z_dim))
        fake_samples = self.var2numpy(self.G(z))
        centers = dg.centers
        print(centers.shape, fake_samples.shape, dg.std)
        l2_store = []
        for s in fake_samples:
            l2_store.append([np.sum((s-c)**2) for c in centers])
        print(len(l2_store))
        mode = np.argmin(l2_store, 1).flatten().tolist()
        print(len(mode))
        dis_ = [l2_store[j][i] for j,i in enumerate(mode)]
        mode_counter = [mode[i] for i in range(len(mode)) if np.sqrt(dis_[i])<=(dg.std*3)]
        print(collections.Counter(mode_counter).values()) 
        print('Number of Modes Captured: ',len(collections.Counter(mode_counter)))
        print('Number of Points Falling Within 3 std. of the Nearest Mode ',sum(collections.Counter(mode_counter).values()))
        #plt.scatter(fake_samples[:, 0], fake_samples[:, 1])
        #plt.savefig('tmp.png')
        fig = plt.figure(figsize=(5, 5))
        modes = centers
        plt.scatter(fake_samples[:, 0], fake_samples[:, 1],
                    s=8, c='g', edgecolor='none', alpha=0.05)
        mog_scale = self.mog_scale
        if self.dataset == 'ring':
            plt.xlim((-1.5 * mog_scale, 1.5 * mog_scale))
            plt.ylim((-1.5 * mog_scale, 1.5 * mog_scale))
        elif self.dataset == 'grid':
            plt.xlim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
            plt.ylim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('tmp.png')
        plt.close(fig)
    
    def test_plot(self):
        if self.dataset == 'grid':
            dg = grid_data_generator()
        elif self.dataset == 'ring':
            dg = data_generator()
        #z = self.tensor2var(torch.randn(self.n_viz, 
        #    self.z_dim))
        #fake_samples = self.var2numpy(self.G(z))
        #fake_samples = self.var2numpy(fake_samples)
        #samples = dg.sample(self.n_viz)
        mog_scale = self.mog_scale
        z = self.tensor2var(torch.randn(self.n_viz, self.z_dim))
        fake_samples = self.var2numpy(self.G(z))
        plt.figure(figsize=(5, 5))
        plt.subplot()
        import seaborn as sns
        ax = sns.kdeplot(fake_samples[:, 0], fake_samples[:, 1], shade=True, \
            cmap='Oranges', n_levels=20, clip=[[-4, 4]]*2, edgecolor='none')
        if self.dataset == 'ring':
            plt.xlim((-1.5 * mog_scale, 1.5 * mog_scale))
            plt.ylim((-1.5 * mog_scale, 1.5 * mog_scale))
        if self.dataset == 'grid':
            modes = dg.centers
            plt.xlim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
            plt.ylim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'fake_denseplot.png'))
        plt.figure(figsize=(5, 5))
        samples = dg.sample(self.n_viz)
        ax = sns.kdeplot(samples[:, 0], samples[:, 1], shade=True, \
            cmap='Oranges', n_levels=20, clip=[[-4, 4]]*2, edgecolor='none')
        if self.dataset == 'ring':
            plt.xlim((-1.5 * mog_scale, 1.5 * mog_scale))
            plt.ylim((-1.5 * mog_scale, 1.5 * mog_scale))
        if self.dataset == 'grid':
            modes = dg.centers
            plt.xlim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
            plt.ylim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'real_denseplot.png')) 
        plt.figure(figsize=(5, 5))
        plt.scatter(samples[:, 0], samples[:, 1], edgecolor='none', alpha=0.05)
        if self.dataset == 'ring':
            plt.xlim((-1.5 * mog_scale, 1.5 * mog_scale))
            plt.ylim((-1.5 * mog_scale, 1.5 * mog_scale))
        if self.dataset == 'grid':
            modes = dg.centers
            plt.xlim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
            plt.ylim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'real_data.png'))
        
    def train_gan_with_csvdd(self):
        if self.dataset == 'grid':
            dg = grid_data_generator()
        elif self.dataset == 'ring':
            dg = data_generator()
        real_label = Variable(torch.ones(self.batch_size)).cuda()
        fake_label = Variable(torch.zeros(self.batch_size)).cuda()
        np_samples_data = dg.sample(self.n_viz)
        mog_scale = self.mog_scale
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(np_samples_data[:, 0], np_samples_data[:, 1],
                    s=8, c='r', edgecolor='none', alpha=0.05)
        if self.dataset == 'ring':
            plt.xlim((-1.5 * mog_scale, 1.5 * mog_scale))
            plt.ylim((-1.5 * mog_scale, 1.5 * mog_scale))
        if self.dataset == 'grid':
            modes = dg.centers
            plt.xlim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
            plt.ylim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.sample_path, 'real.png'))
        plt.close(fig)
        
        if self.pretrained_model == None:
            start = 0
        else:
            start = self.pretrained_model
        for step in range(start, self.total_step):
            self.G.train()
            self.D.train()
            # train on D
            real_samples = dg.sample(self.batch_size)
            real_samples = self.numpy2var(real_samples)
            d_out_real = self.D(real_samples)
            d_loss_real = F.binary_cross_entropy_with_logits(d_out_real, \
                real_label)
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            d_out_fake = self.D(fake_samples)
            d_loss_fake = F.binary_cross_entropy_with_logits(d_out_fake, \
                fake_label)
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            # train on G
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            g_out_fake = self.D(fake_samples)
            g_loss_fake = F.binary_cross_entropy_with_logits(g_out_fake, \
                real_label)
            # support match
            real_samples = dg.sample(self.batch_size)
            real_samples = self.numpy2var(real_samples)
            support_real = self.calculate_csvdd_support(real_samples) 
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            support_fake = self.calculate_csvdd_support(fake_samples)
            support_loss = F.mse_loss(support_real, support_fake)
            #support_loss = -F.cosine_similarity(support_real, support_fake, dim=0)
            g_loss = g_loss_fake*self.lambda_d+support_loss*self.lambda_s
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            print(step, d_loss.data, g_loss.data, support_loss.data)
            if (step+1) % self.sample_step == 0:
                samples = dg.sample(self.n_viz)
                z = self.tensor2var(torch.randn(self.n_viz, 
                    self.z_dim))
                fake_samples = self.G(z)
                #plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                #    edgecolor='none')
                #plt.scatter(fake_samples[:, 0], fake_samples[:, 1], 
                #    c='g', edgecolor='none')
                #plt.axis('off')
                fake_samples = self.var2numpy(fake_samples)
                centers = dg.centers
                l2_store = []
                for s in fake_samples:
                    l2_store.append([np.sum((s-c)**2) for c in centers])
                print(len(l2_store))
                mode = np.argmin(l2_store, 1).flatten().tolist()
                print(len(mode))
                dis_ = [l2_store[j][i] for j,i in enumerate(mode)]
                mode_counter = [mode[i] for i in range(len(mode)) if np.sqrt(dis_[i])<=(dg.std*3)]
                print(collections.Counter(mode_counter).values()) 
                print('Number of Modes Captured: ',len(collections.Counter(mode_counter)))
                print('Number of Points Falling Within 3 std. of the Nearest Mode ',sum(collections.Counter(mode_counter).values()))
                fig = plt.figure(figsize=(5, 5))
                plt.scatter(samples[:, 0], samples[:, 1],
                            s=8, c='r', edgecolor='none', alpha=0.05)
                #plt.scatter(np_samples_est[:, 0], np_samples_est[:, 1],
                #            s=8, c='g', edgecolor='none', alpha=0.05)
                plt.scatter(fake_samples[:, 0], fake_samples[:, 1],
                            s=8, c='g', edgecolor='none', alpha=0.05)
                mog_scale = self.mog_scale
                if self.dataset == 'ring':
                    plt.xlim((-1.5 * mog_scale, 1.5 * mog_scale))
                    plt.ylim((-1.5 * mog_scale, 1.5 * mog_scale))
                elif self.dataset == 'grid':
                    plt.xlim((modes[0][0] - 1.5 * mog_scale,
                              modes[-1][-1] + 1.5 * mog_scale))
                    plt.ylim((modes[0][0] - 1.5 * mog_scale,
                              modes[-1][-1] + 1.5 * mog_scale))
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.sample_path, '{}_fake.png'
                    .format(step+1)))
                plt.close(fig)
            #print(step, support_real.data)
            #print(step, support_fake.data) 
            #sample generator
        # save model
        torch.save({
            'D': self.D.state_dict(),
            'G': self.G.state_dict()
        }, os.path.join(self.model_save_path, '{}.pth'.format \
            (self.total_step)))
        #checkpoint = torch.load(os.path.join(self.model_save_path, \
            #'{}.pth'.format(5000)))
        #self.D.load_state_dict(checkpoint['D'])
        #self.G.load_state_dict(checkpoint['G'])
        # plot G(z)
        #plt.figure(1)
        #z = self.tensor2var(torch.randn(self.batch_size, 
        #            self.z_dim))
        #fake_samples = self.G(z).detach()
        #samples = dg.sample(512)
        #plt.scatter(samples[:, 0], samples[:, 1], c='b', 
        #    edgecolor='none')
        #plt.scatter(fake_samples[:, 0], fake_samples[:, 1], 
        #    c='g', edgecolor='none')
        #plt.axis('off')
        #step = 4999
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(np_samples_data[:, 0], np_samples_data[:, 1],
                    s=8, c='r', edgecolor='none', alpha=0.05)
        #plt.scatter(np_samples_est[:, 0], np_samples_est[:, 1],
        #            s=8, c='g', edgecolor='none', alpha=0.05)
        samples = dg.sample(self.n_viz)
        z = self.tensor2var(torch.randn(self.n_viz, 
            self.z_dim))
        fake_samples = self.G(z).detach()
        plt.scatter(fake_samples[:, 0], fake_samples[:, 1],
                    s=8, c='g', edgecolor='none', alpha=0.05)
        mog_scale = self.mog_scale
        if self.dataset == 'ring':
            plt.xlim((-1.5 * mog_scale, 1.5 * mog_scale))
            plt.ylim((-1.5 * mog_scale, 1.5 * mog_scale))
        if self.dataset == 'grid':
            modes = dg.centers
            plt.xlim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
            plt.ylim((modes[0][0] - 1.5 * mog_scale,
                      modes[-1][-1] + 1.5 * mog_scale))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.sample_path, '{}_fake.png'
            .format(self.total_step)))
        # plot D's boundary
        xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace \
            (-10, 10, 500))
        plt.figure(2)
        xy = self.tensor2var(torch.FloatTensor(np.c_[xx \
        .ravel(), yy.ravel()])) 
        z = self.D(xy)
        zz = self.var2numpy(z)
        zz = zz.reshape(xx.shape)
        ct = plt.contour(xx, yy, zz, linewidths=2)
        plt.clabel(ct, inline=1, fontsize='smaller')
        samples = dg.sample(512)
        plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
        plt.savefig(os.path.join(self.result_path, 'D_boundary.png'))
        plt.close()
        
    def train_ocsvm(self):
        self.ocsvm = One_Class_SVM().cuda()
        self.ocsvm_optimizer = torch.optim.Adam(self.ocsvm.parameters(), \
            self.d_lr, [self.beta1, self.beta2])
        
        dg = data_generator()
        for step in range(30000):
            self.ocsvm.train()
            samples = dg.sample_certain_gauss(512, 4)
            samples = self.tensor2var(torch.FloatTensor(samples))   
            loss, _ = self.ocsvm(samples)
            self.ocsvm_optimizer.zero_grad()
            loss.backward()
            self.ocsvm_optimizer.step()
            print(step, loss)
            
        xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace( \
            -10, 10, 500))
        xy = self.tensor2var(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        _, z = self.ocsvm(xy)
        z = self.var2numpy(z)
        z = z.reshape(xx.shape)
        plt.figure(1)
        ct = plt.contour(xx, yy, z, linewidths=2)
        plt.clabel(ct, inline=1, fontsize='smaller')
        samples = dg.sample(512)
        samples = self.tensor2var(torch.FloatTensor(samples))   
        plt.scatter(samples[:, 0], samples[:, 1], c='b')
        plt.savefig(os.path.join(self.result_path, 'ocsvm_one_gauss.png'))
    
    def train_mocsvm_(self):
        dg = data_generator()
        gamma_new = self.init_gamma().cuda()  
        samples_new = dg.sample(512)
        samples_new = self.tensor2var(torch.FloatTensor(samples_new))    
        old_loss = 0
        
        svm_step = 25000
        start = time.time()
        old_loss = float('inf')
        count_loss = 0
        for step in range(self.pretrained_model, svm_step):
            self.ocsvm.train()
            self.gate.train()
            ## use results of last iter
            samples = samples_new
            gamma = gamma_new
            prior = torch.mean(gamma_new, dim=0, keepdim=True)
            #encoded = fcwta.encode(sess, samples)
            #encoded = self.tensor2var(torch.FloatTensor(encoded))
            #encoded[encoded>0] == 1
            #loss, scores = self.ocsvm(samples, encoded)  
             
            # compute new images and new gamma
            samples_new = dg.sample(512)
            samples_new = self.tensor2var(torch.FloatTensor(samples_new))
            _, softmax, _ = self.gate(samples_new, self.gum_t, True)
            _, l, _, _ = self.ocsvm(samples_new, gamma)
            in_probs = torch.exp(l)
            ds = torch.mul(softmax, in_probs)
            #ds = torch.mul(ds, prior)
            ds_sum = torch.sum(ds, dim=1, keepdim=True)
            ds_sum[ds_sum==0] = 1
            gamma_new = torch.div(ds, ds_sum).detach()
            # train ocsvm
            ocsvm_loss, _, _, expert_loss = self.ocsvm(samples, gamma)
            if (step+1) % 100 == 0:
                new_loss = count_loss/100
                print(step, old_loss, new_loss)
                if old_loss - new_loss < 0.000001:
                    break
                else:
                    old_loss = new_loss
                    count_loss = 0
            self.reset_grad()
            ocsvm_loss.backward(retain_graph=True)
            self.ocsvm_optimizer.step()
            # train gating network
            _, _, logits = self.gate(samples, self.gum_t, True)
            gate_loss = torch.mean(torch.mul(-F.log_softmax(logits, \
                dim=1), gamma))
            
            self.reset_grad()
            gate_loss.backward()
            self.gate_optimizer.step()
            count_loss += expert_loss+gate_loss#+torch.mean(torch. \
                #log(-prior))
            print(step, ocsvm_loss, gate_loss)
        end = time.time()
        print('time is:', end-start)    
        # save model
        torch.save({
            'ocsvm': self.ocsvm.state_dict(),
            'gate': self.gate.state_dict()
        }, os.path.join(self.model_save_path, '{}.pth'.format(step+1)))
        #checkpoint = torch.load(os.path.join(self.model_save_path, \
            #'{}.pth'.format(step+1)))
        #self.ocsvm.load_state_dict(checkpoint['ocsvm'])
        #self.gate.load_state_dict(checkpoint['gate'])
        
        #draw result of support estimation
        xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
        plt.figure(1)
        xy = self.tensor2var(torch.FloatTensor(np.c_[xx \
        .ravel(), yy.ravel()]))
        #encoded = fcwta.encode(sess, xy)
        #encoded = self.tensor2var(torch.FloatTensor(encoded))
        #encoded[encoded>0] == 1
        _, softmax, _ = self.gate(xy, self.gum_t, True)        
        _, _, scores, _ = self.ocsvm(xy, softmax)
        z = torch.mean(torch.mul(scores, softmax), dim=1)
        z = self.var2numpy(z)
        
        # show different boundaries
        for i in range(8):
            samples = dg.sample_certain_gauss(512, i)
            samples = self.tensor2var(torch.FloatTensor(samples))
            _, s, _ = self.gate(samples, self.gum_t, True)
            print(i, torch.mean(s, dim=0))
        for i in range(8):
            plt.figure(i)
            zz = self.var2numpy(scores[:, i])
            print(zz[0])
            zz = zz.reshape(xx.shape)
            ct = plt.contour(xx, yy, zz, levels=[-1, 0, 1], linewidths=2)
            plt.clabel(ct, inline=1, fontsize='smaller')
            samples = dg.sample(512)
            plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                        edgecolor='none')
            plt.savefig(os.path.join(self.result_path, 'mocsvm_{}.png' \
                .format(i)))
            plt.close()
        plt.figure(8)
        z = z.reshape(xx.shape)
        ct = plt.contour(xx, yy, z, linewidths=2)
        plt.clabel(ct, inline=1, fontsize='smaller')
        samples = dg.sample(512)
        plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
        plt.savefig(os.path.join(self.result_path, '{}mocsvm.png'.format(1)))
        plt.close()
        # fit support estimation
        
        for i in range(0):
            self.G.train()
            #train on generator
            real_samples = dg.sample(self.batch_size)
            real_samples = self.tensor2var(torch.FloatTensor(real_samples))
            _, softmax_real, _ = self.gate(real_samples, self.gum_t, True)
            _, svm_out, scores, _ = self.ocsvm(real_samples, softmax_real)
            support_real = torch.mean(torch.mul(scores, softmax_real), dim=0)
            svm_real = support_real/torch.sum(support_real)
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            _, softmax_fake, _ = self.gate(fake_samples, self.gum_t, True)
            _, svm_out, scores, _ = self.ocsvm(fake_samples, softmax_fake)
            support_fake = torch.mean(torch.mul(scores, \
                softmax_fake), dim=0)
            svm_fake = support_fake/torch.sum(support_fake)
            if i==0 or i==1:
                print(svm_real.data)
            g_loss = F.mse_loss(svm_fake, svm_real)
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            print(i, g_loss.data[0])
            
            #sample generator
            if (i+1) % self.sample_step == 0:
                z = self.tensor2var(torch.randn(self.batch_size, 
                    self.z_dim))
                fake_samples = self.G(z).detach()
                plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
                plt.scatter(fake_samples[:, 0], fake_samples[:, 1], 
                    c='g', edgecolor='none')
                plt.axis('off')
                plt.savefig(os.path.join(self.sample_path, '{}_fake.png'
                    .format(i+1)))
                plt.close()
                
        # train gan with support regularization
        real_label = Variable(torch.ones(self.batch_size)).cuda()
        fake_label = Variable(torch.zeros(self.batch_size)).cuda()
        for i in range(0):
            self.G.train()
            self.D.train()
            # train on discriminator
            real_samples = dg.sample(self.batch_size)
            real_samples = self.tensor2var(torch.FloatTensor(real_samples))
            d_out_real = self.D(real_samples)
            d_loss_real = F.binary_cross_entropy_with_logits(d_out_real, \
                real_label)
            #compute loss with fake data
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            d_out_fake = self.D(fake_samples)
            d_loss_fake = F.binary_cross_entropy_with_logits(d_out_fake, \
                fake_label)
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            #train on generator
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            _, softmax, _ = self.gate(fake_samples, self.gum_t, True)
            _, svm_out, scores, _ = self.ocsvm(fake_samples, softmax)
            support_fake = torch.mean(torch.mul(scores, softmax), dim=0)
            svm_fake = support_fake/torch.sum(support_fake)
            _, softmax, _ = self.gate(real_samples, self.gum_t, True)
            _, svm_out, scores, _ = self.ocsvm(real_samples, softmax)
            support_real = torch.mean(torch.mul(scores, softmax), dim=0)
            svm_real = support_real/torch.sum(support_real)
            svm_loss = F.mse_loss(svm_fake, svm_real)
            self.writer.add_histogram('svm_real', svm_real, i)
            self.writer.add_histogram('svm_fake', svm_fake, i)
            g_out_fake = self.D(fake_samples)
            g_loss_fake = F.binary_cross_entropy_with_logits(g_out_fake, \
                real_label)
            g_loss = g_loss_fake+svm_loss*self.lambda_s
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            print(i, d_loss.data[0], svm_loss.data[0], g_loss_fake.data[0])
            self.writer.add_scalar('d_loss', d_loss, i)
            self.writer.add_scalar('g_loss_fake', g_loss_fake, i)
            self.writer.add_scalar('svm_loss', svm_loss, i)
            #self.writer.add_scalar('balance_loss', balance_loss, i)
            #sample generator
            if (i+1) % self.sample_step == 0:
                z = self.tensor2var(torch.randn(self.batch_size, 
                    self.z_dim))
                fake_samples = self.G(z).detach()
                plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
                plt.scatter(fake_samples[:, 0], fake_samples[:, 1], 
                    c='g', edgecolor='none')
                plt.axis('off')
                plt.savefig(os.path.join(self.sample_path, '{}_fake.png'
                    .format(i+1)))
                plt.close()
        # plot D's boundary
        xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace \
            (-10, 10, 500))
        plt.figure(2)
        xy = self.tensor2var(torch.FloatTensor(np.c_[xx \
        .ravel(), yy.ravel()])) 
        z = F.sigmoid(self.D(xy))
        zz = self.var2numpy(z)
        zz = zz.reshape(xx.shape)
        ct = plt.contour(xx, yy, zz, linewidths=2)
        plt.clabel(ct, inline=1, fontsize='smaller')
        samples = dg.sample(512)
        plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
        plt.savefig(os.path.join(self.result_path, 'D_boundary.png'))
        plt.close()
            
    def train_mocsvm(self):
        # sklearn ocsvm
        #classifier = OneClassSVM(kernel='rbf', nu=0.261, gamma=0.05)
        #xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
        #plt.figure(1)
        #dg = data_generator()
        #samples = dg.sample(self.batch_size)
        #classifier = classifier.fit(samples)
        #z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        #z = z.reshape(xx.shape)
        #ct = plt.contour(xx, yy, z, linewidths=2)
        #plt.clabel(ct, inline=1, fontsize='smaller')
        #samples = dg.sample(512)
        #print(classifier.predict([[-4, -4], [0, 0], [0,2], [2, 0], [2, 0.1], [4, 0]]))
        #plt.scatter(samples[:, 0], samples[:, 1], color='b')
        #bbox_args = dict(boxstype='round', fc='0.8')
        #arrow_args = dict(arrowstyle='->')
        #plt.xlim((xx.min(), xx.max()))
        #plt.ylim((yy.min(), yy.max()))
        #plt.savefig(os.path.join(self.result_path, '{}_ocsvm_sklearn.png'.format(1)))
        
        # fcwta_ae
        #fcwta = FullyConnectedWTA(2,
                              #512,
                              #sparsity=0.1,
                              #hidden_units=20,
                              #encode_layers=1,
                              #learning_rate=1e-2)
                              
        #sess = tf.Session()
        #ckpt = tf.train.get_checkpoint_state('train_2019_03_15_21_51_06')
        #fcwta.saver.restore(sess, ckpt.model_checkpoint_path)
        #for i in range(8):
            #samples = dg.sample_certain_gauss(5, i)
            #encoded = fcwta.encode(sess, samples)
            #print(encoded)
            #self.writer.add_histogram('encoded_{}'.format(i), encoded[:, 1], 0)
                   
        dg = data_generator()
        gamma_new = self.init_gamma().cuda()   
        samples_new = dg.sample(512) 
        samples_new = self.tensor2var(torch.FloatTensor(samples_new))    
        old_loss = 0
        
        svm_step = 0
        start = time.time()
        for step in range(svm_step):
            self.ocsvm.train()
            self.gate.train()
            ## use results of last iter
            samples = samples_new
            gamma = gamma_new
            
            #encoded = fcwta.encode(sess, samples)
            #encoded = self.tensor2var(torch.FloatTensor(encoded))
            #encoded[encoded>0] == 1
            #loss, scores = self.ocsvm(samples, encoded)  
             
            # compute new images and new gamma
            samples_new = dg.sample(512)
            samples_new = self.tensor2var(torch.FloatTensor(samples_new))
            _, softmax, _ = self.gate(samples_new, self.gum_t, True)
            _, l, _ = self.ocsvm(samples_new, gamma)
            in_probs = torch.exp(l)
            ds = torch.mul(softmax, in_probs)
            ds_sum = torch.sum(ds, dim=1, keepdim=True)
            ds_sum[ds_sum==0] = 1
            gamma_new = torch.div(ds, ds_sum).detach()
            
            # train ocsvm
            ocsvm_loss, _, _ = self.ocsvm(samples, gamma)
            self.reset_grad()
            ocsvm_loss.backward(retain_graph=True)
            self.ocsvm_optimizer.step()
            ocsvm_loss = self.var2numpy(ocsvm_loss)
            
            # train gating network
            _, _, logits = self.gate(samples, self.gum_t, True)
            gate_loss = torch.mean(torch.mul(-F.log_softmax(logits, \
                dim=1), gamma))
            if gate_loss < 0.0001:
                break
            self.reset_grad()
            gate_loss.backward()
            self.gate_optimizer.step()
            print(i, ocsvm_loss, gate_loss)
        end = time.time()
        print('time is:', end-start)    
        # save model
        #torch.save({
            #'ocsvm': self.ocsvm.state_dict(),
            #'gate': self.gate.state_dict()
        #}, os.path.join(self.model_save_path, '{}.pth'.format(i+1)))
        checkpoint = torch.load(os.path.join(self.model_save_path, \
            '{}.pth'.format(50000)))
        self.ocsvm.load_state_dict(checkpoint['ocsvm'])
        self.gate.load_state_dict(checkpoint['gate'])
        
        #draw result of support estimation
        xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
        plt.figure(1)
        xy = self.tensor2var(torch.FloatTensor(np.c_[xx \
        .ravel(), yy.ravel()]))
        #encoded = fcwta.encode(sess, xy)
        #encoded = self.tensor2var(torch.FloatTensor(encoded))
        #encoded[encoded>0] == 1
        _, softmax, _ = self.gate(xy, self.gum_t, True)        
        _, _, scores = self.ocsvm(xy, softmax)
        z = torch.mean(torch.mul(scores, softmax), dim=1)
        z = self.var2numpy(z)
        #if os.path.exits(os.path.join(self.result_path, 'z.data'):
            #with open(os.path.join(self.result_path, 'z.data'), 'wb') as fp:
                #z = pick.load(fp)
        #with open(os.path.join(self.result_path, 'z.data'), 'wb') as fp:
            #pickle.dump(z, fp)
        # show different boundaries
        for i in range(8):
            samples = dg.sample_certain_gauss(512, i)
            samples = self.tensor2var(torch.FloatTensor(samples))
            _, s, _ = self.gate(samples, self.gum_t, True)
            print(i, torch.mean(s, dim=0))
        for i in range(8):
            plt.figure(i)
            zz = self.var2numpy(scores[:, i])
            print(zz[0])
            zz = zz.reshape(xx.shape)
            ct = plt.contour(xx, yy, zz, levels=[-1, 0, 1], linewidths=2)
            plt.clabel(ct, inline=1, fontsize='smaller')
            samples = dg.sample(512)
            plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                        edgecolor='none')
            plt.savefig(os.path.join(self.result_path, 'mocsvm_{}.png' \
                .format(i)))
        plt.figure(8)
        z = z.reshape(xx.shape)
        ct = plt.contour(xx, yy, z, linewidths=2)
        plt.clabel(ct, inline=1, fontsize='smaller')
        samples = dg.sample(512)
        plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
        plt.savefig(os.path.join(self.result_path, '{}mocsvm.png'.format(1)))
        # fit support estimation
        for i in range(25000):
            self.G.train()
            #train on generator
            
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            _, softmax, _ = self.gate(fake_samples, self.gum_t, True)
            _, svm_out, _ = self.ocsvm(fake_samples, softmax)
            svm_loss = torch.mean(torch.mul(-svm_out, softmax))
            fake_gate = softmax.sum(dim=0)/softmax.sum()
            real_samples = dg.sample(512)
            real_samples = self.tensor2var(torch.FloatTensor(real_samples))
            _, s, _ = self.gate(real_samples, self.gum_t, True)
            gate_prior = s.sum(dim=0)/s.sum()
            balance_loss = F.mse_loss(fake_gate, gate_prior)
            g_loss = svm_loss + balance_loss*0
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            print(i, svm_loss.data[0], balance_loss.data[0])
            
            #sample generator
            if (i+1) % self.sample_step == 0:
                z = self.tensor2var(torch.randn(self.batch_size, 
                    self.z_dim))
                fake_samples = self.G(z).detach()
                plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
                plt.scatter(fake_samples[:, 0], fake_samples[:, 1], 
                    c='g', edgecolor='none')
                plt.axis('off')
                plt.savefig(os.path.join(self.sample_path, '{}_fake.png'
                    .format(i+1)))
                plt.close()
                
        # train gan with support regularization
        real_label = Variable(torch.ones(self.batch_size)).cuda()
        fake_label = Variable(torch.zeros(self.batch_size)).cuda()
        for i in range(0):
            self.G.train()
            self.D.train()
            # train on discriminator
            real_samples = dg.sample(self.batch_size)
            real_samples = self.tensor2var(torch.FloatTensor(real_samples))
            d_out_real = self.D(real_samples)
            d_loss_real = F.binary_cross_entropy_with_logits(d_out_real, \
                real_label)
            #compute loss with fake data
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            d_out_fake = self.D(fake_samples)
            d_loss_fake = F.binary_cross_entropy_with_logits(d_out_fake, \
                fake_label)
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            #train on generator
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            _, softmax, _ = self.gate(fake_samples, self.gum_t, True)
            _, svm_out, _ = self.ocsvm(fake_samples, softmax)
            svm_loss = torch.mean(torch.mul(-svm_out, softmax))
            g_out_fake = self.D(fake_samples)
            g_loss_fake = F.binary_cross_entropy_with_logits(g_out_fake, \
                real_label)
            fake_gate = softmax.sum(dim=0)/softmax.sum()
            _, s, _ = self.gate(real_samples, self.gum_t, True)
            gate_prior = s.sum(dim=0)/s.sum()
            balance_loss = F.mse_loss(fake_gate, gate_prior)
            g_loss = g_loss_fake+svm_loss+balance_loss*0
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            print(i, d_loss.data[0], svm_loss.data[0], g_loss_fake.data[0], balance_loss.data[0])
            self.writer.add_scalar('d_loss', d_loss, i)
            self.writer.add_scalar('g_loss_fake', g_loss_fake, i)
            self.writer.add_scalar('svm_loss', svm_loss, i)
            self.writer.add_scalar('balance_loss', balance_loss, i)
            #sample generator
            if (i+1) % self.sample_step == 0:
                z = self.tensor2var(torch.randn(self.batch_size, 
                    self.z_dim))
                fake_samples = self.G(z).detach()
                plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
                plt.scatter(fake_samples[:, 0], fake_samples[:, 1], 
                    c='g', edgecolor='none')
                plt.axis('off')
                plt.savefig(os.path.join(self.sample_path, '{}_fake.png'
                    .format(i+1)))
                plt.close()
    
    def gate_out(self, clusters, x):
        i = 0
        c = clusters[i].repeat(x.size(0), 1)
        dist_to_c = torch.norm((x-c), p=2, dim=1).unsqueeze(1)
        for i in range(1, clusters.size(0)):
            c = clusters[i].repeat(x.size(0), 1)
            temp = torch.norm((x-c), p=2, dim=1).unsqueeze(1)
            dist_to_c = torch.cat([dist_to_c, temp], dim=1)
        g = torch.softmax(-dist_to_c, dim=1)
        max_idx = torch.argmax(g, dim=1, keepdim=True)
        max_idx = self.var2tensor(max_idx)
        one_hot = torch.FloatTensor(g.size())
        one_hot.zero_()
        one_hot.scatter_(1, max_idx, 1)
        one_hot = self.tensor2var(one_hot)
        return one_hot
        
    def train_kmocsvm(self):
        dg = data_generator()
        samples = dg.sample(1000)
        kmeans = KMeans(n_clusters=8, random_state=0).fit(samples)
        clusters = kmeans.cluster_centers_
        clusters = self.tensor2var(torch.FloatTensor(clusters))
        print(clusters.data)
        total_step = 25000
        for i in range(total_step):
            samples = dg.sample(512)
            samples = self.tensor2var(torch.FloatTensor(samples))   
            g = self.gate_out(clusters, samples)         
            svm_loss, _, _, _ = self.ocsvm(samples, g)
            self.reset_grad()
            svm_loss.backward()
            self.ocsvm_optimizer.step()
            print(svm_loss)
        # save model
        torch.save({
            'ocsvm': self.ocsvm.state_dict()
        }, os.path.join(self.model_save_path, '{}.pth'.format(total_step)))
        #checkpoint = torch.load(os.path.join(self.model_save_path, \
           #'{}.pth'.format(i+1)))
        #self.ocsvm.load_state_dict(checkpoint['ocsvm'])
        #self.gate.load_state_dict(checkpoint['gate'])
        #draw result of support estimation
        xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
        
        xy = self.tensor2var(torch.FloatTensor(np.c_[xx \
        .ravel(), yy.ravel()]))
        g = self.gate_out(clusters, xy)  
        _, _, scores, _ = self.ocsvm(xy, g)
        z = torch.mean(torch.mul(scores, g), dim=1)
        z = self.var2numpy(z)
        
        # show weight of experts
        for i in range(8):
            samples = dg.sample_certain_gauss(512, i)
            samples = self.tensor2var(torch.FloatTensor(samples))
            s = self.gate_out(clusters, samples)
            print(i, torch.mean(s, dim=0))
        
        # show different boundaries
        for i in range(8):
            plt.figure(i)
            zz = self.var2numpy(scores[:, i])
            print(zz[0])
            zz = zz.reshape(xx.shape)
            ct = plt.contour(xx, yy, zz, levels=[-1, 0, 1], linewidths=2)
            plt.clabel(ct, inline=1, fontsize='smaller')
            samples = dg.sample(512)
            plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                        edgecolor='none')
            plt.savefig(os.path.join(self.result_path, 'kmocsvm_{}.png' \
                .format(i)))
        plt.figure(8)
        z = z.reshape(xx.shape)
        ct = plt.contour(xx, yy, z, linewidths=2)
        plt.clabel(ct, inline=1, fontsize='smaller')
        samples = dg.sample(512)
        plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
        plt.savefig(os.path.join(self.result_path, 'kmocsvm.png'.format(1)))
        plt.close()

    def train_kmeans(self):
        
        # mode detection using kmeans
        dg = data_generator()
        samples = dg.sample(1000)
        kmeans = KMeans(n_clusters=8, random_state=0).fit(samples)
        clusters = kmeans.cluster_centers_
        for step in range(25000):
            self.gate.train()
            samples = dg.sample(512)
            samples_var = self.tensor2var(torch.FloatTensor(samples))
            weights, _, _ = self.gate(samples_var, self.gum_t, True)
            weights = weights.cuda()
            to_clusters = self.loss_function(samples_var, clusters)
            real_loss = torch.mean(torch.mul(to_clusters, weights))
            self.reset_grad()
            real_loss.backward()
            self.gate_optimizer.step()
        target = (weights.sum(dim=0)/weights.sum()).detach()
        print(target)
        print(weights[:10])
        xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5,\
            5, 500))
        zz_to_clusters = self.loss_function(self.tensor2var(torch. \
            FloatTensor(np.c_[xx.ravel(), yy.ravel()])), clusters)
        zz_weights, _, _ = self.gate(self.tensor2var(torch.FloatTensor \
            (np.c_[xx.ravel(), yy.ravel()])), self.gum_t, True)
        zz = self.var2numpy(torch.sum(torch.mul(zz_to_clusters, \
            zz_weights),dim=1))
        zz = zz.reshape(xx.shape)
        plt.figure(1)
        plt.scatter(samples[:, 0], samples[:, 1], color='b')
        ct = plt.contour(xx, yy, zz, linewidths=2)
        plt.clabel(ct, inline=1, fontsize='smaller')
        plt.xlim((xx.min(), xx.max()))
        plt.ylim((yy.min(), yy.max()))
        plt.legend()
        plt.savefig(os.path.join(self.result_path, '{}_kmeans.png'. \
            format(1)))
        plt.close()
            
        #plt.scatter(clusters[:, 0], clusters[:, 1])
        real_label = Variable(torch.ones(self.batch_size)).cuda()
        fake_label = Variable(torch.zeros(self.batch_size)).cuda()
        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)

        score_step = self.score_epoch * step_per_epoch
        score_start = self.score_start * step_per_epoch
        start_anneal = self.start_anneal * step_per_epoch
        step_t_decay = self.step_t_decay * step_per_epoch
        self.balance_w_start = copy.deepcopy(self.balance_weight)
        for step in range(self.total_step):
            self.G.train()
            self.D.train()
            #train on discriminator
            #compute loss with real data
            real_samples = dg.sample(self.batch_size)
            real_samples = self.tensor2var(torch.FloatTensor(real_samples))
            d_out_real = self.D(real_samples)
            d_loss_real = F.binary_cross_entropy_with_logits(d_out_real, \
                real_label)
            #compute loss with fake data
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            d_out_fake = self.D(fake_samples)
            d_loss_fake = F.binary_cross_entropy_with_logits(d_out_fake, \
                fake_label)
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            # train on generator
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            fake_samples = self.G(z)
            g_out_fake = self.D(fake_samples)
            g_loss_fake = F.binary_cross_entropy_with_logits(g_out_fake, \
                real_label)*0
            #z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            #fake_samples = self.G(z)
            weights, _, _ = self.gate(fake_samples, self.gum_t, True)
            weights = weights.cuda()
            to_clusters = self.loss_function(fake_samples, clusters)
            fake_loss = torch.mean(torch.mul(to_clusters, weights))
            weights_k = weights.sum(dim=0)/weights.sum()
            #target = Variable(torch.ones(self.num_dis)/self.num_dis).cuda()
            balance_loss = F.mse_loss(weights_k, target)*10
            g_loss = g_loss_fake + (fake_loss+balance_loss)*self.balance_weight
            self.reset_grad()
            
            grad = torch.autograd.grad(outputs=balance_loss,
                                           inputs=self.G.parameters(),
                                           grad_outputs=torch.ones(g_loss.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]
            b_grad_l2norm = torch.sqrt(torch.sum(grad**2))
            
            grad = torch.autograd.grad(outputs=fake_loss,
                                           inputs=self.G.parameters(),
                                           grad_outputs=torch.ones(g_loss.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]
            f_grad_l2norm = torch.sqrt(torch.sum(grad**2))
            grad = torch.autograd.grad(outputs=g_loss_fake,
                                           inputs=self.G.parameters(),
                                           grad_outputs=torch.ones(g_loss.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]
            g_grad_l2norm = torch.sqrt(torch.sum(grad**2))
            
            
            g_loss.backward()
            self.g_optimizer.step()
            #self.writer.add_histogram('g_loss_grad', grad_l2norm, step)
            if self.use_tensorboard == True:
                self.writer.add_scalar('d_loss', d_loss, step)
                self.writer.add_scalar('g_loss_fake', g_loss_fake, step)
                self.writer.add_scalar('fake_loss', fake_loss, step)
                self.writer.add_scalar('balance_loss', balance_loss, step)
            if (step+1) >= start_anneal and (step+1) % (step_t_decay) == 0:
                self.balance_weight = self.balance_w_start * np.exp \
                    (-self.gum_t_decay*((step+1)-start_anneal))      
                #self.balance_weight = max(self.balance_weight, 0.001) #orignial is 0.0001
                print('self.balance_weight changed by : ', np.exp \
                    (-self.gum_t_decay*((step+1)-start_anneal)),\
                    ' now balance : ', self.balance_weight )
            if (step+1) % self.sample_step == 0:
                z = self.tensor2var(torch.randn(self.batch_size, 
                    self.z_dim))
                fake_samples = self.G(z).detach()
                plt.scatter(samples[:, 0], samples[:, 1], c='b', 
                    edgecolor='none')
                plt.scatter(fake_samples[:, 0], fake_samples[:, 1], 
                    c='g', edgecolor='none')
                plt.axis('off')
                plt.savefig(os.path.join(self.sample_path, '{}_fake.png'
                    .format(step+1)))
                plt.close()
                #print(grad_l2norm)
                print('balance_grad', b_grad_l2norm)
                print('fake_grad', f_grad_l2norm)
                print('g_grad', g_grad_l2norm)
                print(weights_k.data)
    
    #def train(self):
        #dg = data_generator()
        #samples = dg.sample(1000)
        #classifier = OneClassSVM(kernel='rbf', nu=0.261, gamma=0.05)
        #classifier.fit(samples)
        #for step in range(250000):
            #self.G.train()
            
            #z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            #fake_samples = self.G(z)
            #g_loss = -classifier.decision_function(fake_samples

    def test_2(self):
        from mixture_gaussian import data_generator
        dg = data_generator()
        real_images_new = torch.FloatTensor(dg.sample(self.batch_size))
        real_images = torch.FloatTensor(dg.sample_certain_gauss(self.batch_size, 1))
        #z_new_ = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
        #real_images_new = self.G(z_new_).detach()
        plt.scatter(real_images[:, 0], real_images[:, 1], c='g')
        plt.scatter(real_images_new[:, 0], real_images_new[:, 1], c='b')
        plt.savefig(os.path.join(self.sample_path, '{}.png'.format(1)))
        
    def test_moe(self):
        
        # output partition of moe
        from mixture_gaussian import data_generator
        from matplotlib import colors
        self.load_pretrained_model()
        
        colors = list(colors.BASE_COLORS.keys())
        for i in range(1):
            dg = data_generator()
            #real_images_new = torch.FloatTensor(dg.sample_certain_gauss(self.batch_size, i))
            #real_images_new = self.tensor2var(real_images_new)
            z_new_ = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
            real_images_new = self.G(z_new_)
            _, softmax, _ = self.gate(real_images_new, self.gum_t, True)
            d_out_many = self.D(real_images_new)
            label = Variable(torch.ones(self.batch_size, self.num_dis)).cuda()
            d_out = prob_with_logits(d_out_many, label)
            ds = torch.mul(softmax, d_out)
            ds_sum = torch.sum(ds, dim=1, keepdim=True)
            ds_sum[ds_sum==0]=1
            gamma_d_new = torch.div(ds, ds_sum).detach()
            partition = torch.max(gamma_d_new, dim=1)[1]
            
            for j in range(self.num_dis):
                print(i, self.var2numpy(torch.sum(partition==j).data))
        
        #for j in range(samples.size(0)):
        #plt.scatter(samples[:, 0], samples[:, 1], c=partition)
        #plt.savefig(os.path.join(self.sample_path, '{}_partition.png'.format(1)))  
    
    def test_svm(self):
        
        # oneclasssvm
        from sklearn.svm import OneClassSVM
        import warnings
        warnings.filterwarnings("ignore", "Solver terminated early.*")
        dg = data_generator()
        classifiers = []
        for i in range(8):          
            classifier = OneClassSVM(kernel='linear', nu=0.261, gamma=0.05)
            classifiers.append(classifier)
        xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
        plt.figure(1)
               
        for j in range(8):
            samples = dg.sample_certain_gauss(self.batch_size, j)
            classifiers[j] = classifiers[j].fit(samples)
            z = classifiers[j].decision_function(np.c_[xx.ravel(), yy.ravel()])
            z = z.reshape(xx.shape)
            legend = plt.contour(xx, yy, z, levels=[0], linewidths=2)
        
        samples = dg.sample(512)
        print(classifiers[0].predict([[-4, -4], [0, 0], [0,2], [2, 0], [2, 0.1], [4, 0]]))
        #print(classifiers[0].decision_function([[-4, -4], [0, 0], [0,2], [2, 0], [2, 0.1], [4, 0]]))
        #print(classifiers[0].score_samples([[-4, -4], [0, 0], [0,2], [2, 0], [2, 0.1], [4, 0]]))
        plt.scatter(samples[:, 0], samples[:, 1], color='b')
        bbox_args = dict(boxstype='round', fc='0.8')
        arrow_args = dict(arrowstyle='->')
        #plt.annotate('several confoun
        plt.xlim((xx.min(), xx.max()))
        plt.ylim((yy.min(), yy.max()))
        #plt.legend()
        plt.savefig(os.path.join(self.sample_path, '{}_svms.png'.format(1)))
        
            
    def test(self):

        # self.load_pretrained_model()
        # if self.dataset == 'cifar':
        #     print("calculating inception score for step %d...)" % (self.pretrained_model + 1))
        #     print(self.gum_t)
        #     print_inception_score(self.G, self.Gum_D, self.z_dim, self.num_gen, self.image_size, self.gum_t)
        # if self.dataset == 'CelebA' or self.dataset=='LSUN':
        #     print("calculating MS-SSIM score for step %d...)" % (self.pretrained_model + 1))
        #     print_msssim_score(self.G, self.z_dim, self.ms_num_image, self.image_size, self.gum_t)
        #self.load_pretrained_model()
        """print inception scores"""
        #for param in (self.G.parameters()):
        #    print(type(param.data), param.size())
        #if self.dataset == 'cifar':
        #    print("calculating inception score for step %d...)" % (self.pretrained_model + 1))
        #    print(self.gum_t)
        #    print_inception_score(self.G, self.D, self.Gum_D, self.z_dim, self.num_dis, self.image_size, self.gum_t)
        #if self.dataset == 'CelebA' or self.dataset == 'LSUN':
        #    print("calculating MS-SSIM score for step %d...)" % (self.pretrained_model + 1))
        #    print_msssim_score(self.G, self.z_dim, self.ms_num_image, self.image_size, self.gum_t)

        #"""GN output for particular certain gaussian data points"""
        from mixture_gaussian import data_generator
        #import scipy
        #dg = data_generator()
        #for i in range(8):
            #samples = dg.sample_certain_gauss(self.batch_size, i)
            #samples = torch.FloatTensor(samples)
            #samples = self.tensor2var(samples)
            #_, softmax_out, _ = self.gate(samples, self.gum_t, True)
            #print(torch.mean(softmax_out, dim=0))
            #for j in range(self.num_dis):
                #self.writer.add_histogram('hist_{}/{}'.format(i, j), softmax_out[:, j], 0)
        
        
        # oneclasssvm
        from sklearn.svm import OneClassSVM
        import warnings
        warnings.filterwarnings("ignore", "Solver terminated early.*")
        dg = data_generator()
        classifiers = []
        for i in range(self.num_dis):            
            classifier = OneClassSVM(kernel='rbf', nu=0.261, gamma=0.05, max_iter=1)
            classifiers.append(classifier)
        gamma_d_new = self.init_gamma()
        
        for i in range(1000):
            gamma_d = gamma_d_new
            #gamma_d_new = 
            samples = dg.sample(self.batch_size)
            dists = []
            for j in range(self.num_dis):
                classifiers[j] = classifier[j].fit(samples, sample_weight=gamma_d[:, j].view(gamma_d.size(0)).numpy())
                dist = classifier
        xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
        plt.figure(1)
        z = classifer.decision_function(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        samples = dg.sample(512)
        legend = plt.contour(xx, yy, z, levels=[0], linewidths=2)
        print(classifer.predict([[-4, -4], [0, 0], [0,2], [0, 3]]))
        plt.scatter(samples[:, 0], samples[:, 1], color='b')
        bbox_args = dict(boxstype='round', fc='0.8')
        arrow_args = dict(arrowstyle='->')
        #plt.annotate('several confoun
        plt.xlim((xx.min(), xx.max()))
        plt.ylim((yy.min(), yy.max()))
        #plt.legend()
        plt.savefig(os.path.join(self.sample_path, '{}_svm.png'.format(1)))
        """gate out for real data"""
        #samples = dg.sample(self.batch_size)
        #samples = torch.FloatTensor(samples)
        #samples = self.tensor2var(samples)
        #gate_out_real, softmax_out, _ = self.gate(samples, self.gum_t, True)
        #d_out_real_many = self.D(samples)
        #d_out_real = torch.sum(torch.mul(d_out_real_many, gate_out_real), dim=1)
        #xdata = samples[:, 0]
        #ydata = samples[:, 1]
        #zdata = d_out_real
        #ax = plt.axes(projection='3d')
        #ax.scatter3D(xdata, ydata, zdata, c='b')
        ##print(torch.mean(softmax_out, dim=0))
        ##print(softmax_out)
        #"""gate out for fake data"""
        #z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
        #fakes = self.G(z)
        #d_out_fake_many = self.D(fakes)
        #gate_out_fake, softmax_out, _ = self.gate(fakes, self.gum_t, True)
        #d_out_fake = torch.sum(torch.mul(d_out_fake_many, gate_out_fake), dim=1)
        #xdata = fakes[:, 0]
        #ydata = fakes[:, 1]
        #zdata = d_out_fake
        #
        #ax.scatter3D(xdata, ydata, zdata, c='g')
        #print(torch.mean(softmax_out, dim=0))
        #print(softmax_out)
        #data_iter = iter(self.data_loader)
        #real_images, _ = next(data_iter)
        #bs = self.batch_size
        #"""pairwise distance"""
        #real_dists = scipy.spatial.distance.pdist(real_images.view(bs, -1).numpy())
        #self.writer.add_histogram('pdists', real_dists, 0)
        #for i in range(5):
        #    torch.manual_seed(999)
        #    z = self.tensor2var(torch.randn(bs, self.z_dim))
        #    fake_images = self.G(z)
        #    fake_dists = scipy.spatial.distance.pdist(self.var2numpy(fake_images.view(bs, -1)))
        #    self.writer.add_histogram('pdists', fake_dists, i+1)
            
        #data_iter = iter(self.data_loader)
        #real_images, _ = next(data_iter)
        #real_images = self.tensor2var(real_images)
        #bs = self.batch_size
        #_, softmax_out, _ = self.gate(real_images,self.gum_t,True)
        #for j in range(self.num_dis):
        #    self.writer.add_histogram('hist_{}/{}'.format(0, j), softmax_out[:, j], 0)
        #print(softmax_out[:10])
        #z = self.tensor2var(torch.randn(bs, self.z_dim))
        #fake_images = self.G(z)
        #_, softmax_out, _ = self.gate(fake_images,self.gum_t,True)
        #for j in range(self.num_dis):
        #    self.writer.add_histogram('hist_{}/{}'.format(1, j), softmax_out[:, j], 0)
        #print(softmax_out[:10])
        # import pickle
        # import matplotlib.pyplot as plt
        # import os
        
        # with open(os.path.join(self.result_path, 'result.data'), 'rb') as fp:
        #     loss = pickle.load(fp)
        
        # plt.subplot(311)
        # plt.plot(loss[0], color='b', label='d_loss_real')
        # plt.subplot(312)
        # plt.plot(loss[1], color='r', label='d_loss_fake')
        # plt.subplot(313)
        # plt.plot(loss[2], color='y', label='g_loss_fake')
        # plt.savefig(os.path.join(self.result_path, '{}_result_all.png'.format('cifar')))
