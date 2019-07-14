import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler

import os
from mixture_gaussian import data_generator
from mixture_gaussian import grid_data_generator

import numpy as np


def get_loader(image_path, image_size, dataset, batch_size, num_workers=2):
    """Build and return data loader."""
    if dataset == 'ring':
        dg = data_generator()
        dataloader = dg.sample(batch_size)
        return torch.FloatTensor(dataloader)

    if dataset == 'grid':
        dg = grid_data_generator()
        dataloader = dg.sample(batch_size)
        return torch.FloatTensor(dataloader)

    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transform = transforms.Compose([
            #transforms.Resize((244, 244)),
            #transforms.ToTensor(),
            #transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                 #std=(0.229, 0.224, 0.225))])
    if dataset == 'LSUN':
        dataset = datasets.LSUN(image_path, classes=['church_outdoor_train'], transform=transform)
        # 'church_outdoor' is one category of LSUN dataset
    elif dataset == 'CelebA_FD':
        dataset = datasets.ImageFolder(image_path, transform=transform)
    elif dataset == 'cifar':
        dataset = datasets.CIFAR10('../../data/cifar-10', transform=transform, download=True)
    elif dataset == 'CelebA':
        transform = transforms.Compose([transforms.CenterCrop(160),
                                       transforms.Resize((image_size, image_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_path = os.path.join(image_path, dataset)
        if not os.path.exists(os.path.join(image_path, dataset)):
            os.makedirs(image_path)
        dataset = datasets.ImageFolder(image_path,transform=transform)
    elif dataset == 'mnist':
        dataset = datasets.MNIST('../../data/mnist', train=True, download=True, transform=
                                transforms.Compose([transforms.Pad(2), transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (0.5,))]))
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2, drop_last=True)

    return data_loader

def denorm(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
        
def get_data(normal_class=[3], anomaly_class=[5, 6, 7], anomaly_count=100):
    image_path = '../../data/cifar-10'
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transform = transforms.Compose([transforms.Resize((244, 244)),
        #transforms.ToTensor(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             #std=(0.229, 0.224, 0.225))])
    dataset = datasets.CIFAR10(image_path, transform=transform, download=False)
    # normal data
    normal_set = []
    for c in normal_class:
        (indexes,) = np.where(np.array(dataset.train_labels) == c)
        normal_set.append(indexes)
    normal_indexes = np.concatenate(normal_set, axis=0).reshape(-1).tolist()
    # anomaly data
    anomaly_set = []
    for c in anomaly_class:
        (indexes,) = np.where(np.array(dataset.train_labels) == c)
        anomaly_set.append(indexes[:100])
    anomaly_indexes = np.concatenate(anomaly_set, axis=0).reshape( \
        -1).tolist()
    # load train data
    train_indexes = normal_indexes[:int(len(normal_indexes)/2)]
    train_data_loader = torch.utils.data.DataLoader(dataset=dataset, \
        batch_size=64, sampler = SubsetRandomSampler(train_indexes), \
        shuffle=False, num_workers=2, drop_last=True)
    #data_iter = iter(train_data_loader)
    #images, _ = next(data_iter)
    #save_image(denorm(torch.FloatTensor(images).view(-1, 3, 32, 32)), \
        #'train.png')
    # load test data
    test_indexes = normal_indexes[int(len(normal_indexes)/2): ] + \
        anomaly_indexes
    print(len(test_indexes))
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset, \
        batch_size=64, sampler = SubsetRandomSampler( \
        test_indexes), shuffle=False, num_workers=2, drop_last=True)
    #data_iter = iter(test_data_loader)
    #images, _ = next(data_iter)
    #save_image(denorm(torch.FloatTensor(images).view(-1, 3, 32, 32)), \
        #'test.png')
    return train_data_loader, test_data_loader

def get_mnist(normal_class=[4], anomaly_class=[0, 7, 9]):
    image_path = '../../data/mnist'
    transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(image_path, transform=transform, \
        download=False)
    # normal data
    normal_set = []
    for c in normal_class:
        (indexes,) = np.where(np.array(dataset.train_labels) == c)
        normal_set.append(indexes)
    normal_indexes = np.concatenate(normal_set, axis=0).reshape(-1).tolist()
    # anomaly data
    anomaly_set = []
    for c in anomaly_class:
        (indexes,) = np.where(np.array(dataset.train_labels) == c)
        anomaly_set.append(indexes[:100])
    anomaly_indexes = np.concatenate(anomaly_set, axis=0).reshape( \
        -1).tolist()
    # load train data
    train_indexes = normal_indexes[: int(len(normal_indexes)/2)]
    train_data_loader = torch.utils.data.DataLoader(dataset=dataset, \
        batch_size=32, sampler = SubsetRandomSampler(train_indexes), \
        shuffle=False, num_workers=2, drop_last=True)
    data_iter = iter(train_data_loader)
    images, _ = next(data_iter)
    save_image(denorm(torch.FloatTensor(images)), \
        'train.png')
    # load test data
    test_indexes = normal_indexes[int(len(normal_indexes)/2): ] + \
        anomaly_indexes
    print(len(test_indexes))
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset, \
        batch_size=32, sampler = SubsetRandomSampler( \
        test_indexes), shuffle=False, num_workers=2, drop_last=True)
    data_iter = iter(test_data_loader)
    images, _ = next(data_iter)
    save_image(denorm(torch.FloatTensor(images)), \
        'test.png')
    return train_data_loader, test_data_loader

def main():
    get_mnist()

if __name__ == '__main__':
    main()        
