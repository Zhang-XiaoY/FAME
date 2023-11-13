# import normal package
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import configparser as cp

# torch function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# # import self-defined net structure
# from net_struct.ResNet import ResNet18

# config reading
configs = cp.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config.ini')
configs.read(config_path)

# data and model save setting
data_root_path = configs.get('DEFAULT', 'data_root_path') + '/data'
model_save_path = configs.get('DEFAULT', 'model_save_path') \
                  + '/' +datetime.datetime.now().strftime('%Y%m%d') \
                  + '_' + configs.get('DEFAULT', 'model')
model_name = 'ori_' + configs.get('DEFAULT', 'model')
en_nonidealities = [configs.getint('DEFAULT', 'en_SAF'),
                    configs.getint('DEFAULT', 'en_ThermalNoise'),
                    configs.getint('DEFAULT', 'en_ShotNoise'),
                    configs.getint('DEFAULT', 'en_rtn'),
                    configs.getint('DEFAULT', 'en_prog_error'),]

data_type = getattr(torch, configs.get('DEFAULT', 'precision'))

def simple_train(model,
                 optimizer,
                 scheduler,
                 train_set,
                 test_loader,
                 epoch_num,
                 criterion = nn.CrossEntropyLoss(),
                 k = 10,
                 batch_size = 100,
                 device = torch.device("cuda:0")):
    '''
    # this function implements simple model training
    # if nonideality added, it must be added before this function
    
    # parameters:
    # model: the model to be trained
    # optimizer: optimizer of the model
    # scheduler: learning rate scheduler of the model
    # train_set: training data set, train data will be separated into train and val set
    # test_loader: test data loader
    # epoch_num: number of epochs
    # criterion: loss function, default is CrossEntropyLoss
    # k: number of batches to be used for validation, default is 10
    # device: device to be used, default is cuda:0
    '''

    # if model is not on device, move it to device
    if next(model.parameters()).device != device:
        model.to(device)
    
    # check model save path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    nonideal_model_name = nonideal_nameAdd(model_name, en_nonidealities)
    
    # 3 data set loss and acc array list
    epoch_train_loss = np.zeros(epoch_num)
    epoch_val_loss = np.zeros(epoch_num)
    epoch_test_loss = np.zeros(epoch_num)
    # remove train acc for simplifier
    # epoch_train_acc = np.zeros(epoch_num)
    epoch_val_acc = np.zeros(epoch_num)
    epoch_test_acc = np.zeros(epoch_num)
    epoch_lr = np.zeros(epoch_num)
    
    best_val_loss = 10.0
    
    print('==================== Start Training ====================')
    print('Model: {}\nData precision: {}'.format(
        configs.get('DEFAULT', 'model'),
        configs.get('DEFAULT', 'precision')))
    print('========================================================')
    
    for epoch in range(epoch_num):
        
        train_loss = 0.0
        
        model.train()
        
        # randomly choose valset every epoch
        train_sampler, val_sampler = train_set_split(train_set, k)
        train_loader = DataLoader(train_set, batch_size = batch_size, sampler = train_sampler, num_workers = 4)
        val_loader = DataLoader(train_set, batch_size = batch_size, sampler = val_sampler, num_workers = 4)
        epoch_lr[epoch] = optimizer.param_groups[0]['lr']
        
        for i , data in enumerate(train_loader):
            inputs, labels = data[0].type(data_type).to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # train set loss and acc
        epoch_train_loss[epoch] = train_loss * batch_size / len(train_loader.sampler)
        
        # validation
        epoch_val_loss[epoch], epoch_val_acc[epoch] = simple_test(model, val_loader, criterion, device)
        
        # test
        epoch_test_loss[epoch], epoch_test_acc[epoch] = simple_test(model, test_loader, criterion, device)
        
        # lr scheduler
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss[epoch])
        else:
            scheduler.step()
        
        # print result every 10 epochs
        # if (epoch + 1) % 10 == 0:
        print('[{}][{: >3}/{}] Val Loss: {:.4f}\tTest Acc: {:.2f}%'.format(
            datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
            epoch+1, epoch_num, epoch_val_loss[epoch], (epoch_test_acc[epoch]*100)))
        # if val loss is the best, save model
        if (epoch_val_loss[epoch] < best_val_loss):
            best_val_loss = epoch_val_loss[epoch]
            torch.save(model, model_save_path + '/' + nonideal_model_name + '.pth')
    
    # finish training, save loss and acc
    model_result_save(nonideal_model_name,
                      model_save_path,
                      epoch_train_loss, 
                      epoch_val_loss, 
                      epoch_test_loss, 
                      epoch_val_acc, 
                      epoch_test_acc,
                      epoch_lr)


def simple_test(model, 
                test_loader, 
                criterion = nn.CrossEntropyLoss(), 
                device = torch.device("cuda:0")):
    '''
    # this function implements simple model testing
    
    # parameters:
    # model: the model to be tested
    # test_loader: test data loader
    # device: device to be used, default is cuda:0
    '''
    
    # data set loss and acc array list
    test_loss = 0.0
    test_acc = 0.0
    
    model.eval()
    
    # testing
    for i , data in enumerate(test_loader):
        inputs, labels = data[0].to(device).type(data_type), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_acc += (predicted == labels).sum().item()
    
    test_loss = test_loss * test_loader.batch_size / len(test_loader.sampler)
    test_acc /= len(test_loader.sampler)
    
    return test_loss, test_acc
    
def train_set_split(train_set, k):
    '''
    # this function split train set into train and val set
    # val set has 1/k data of ori train set
    ''' 
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(num_train/k))
    train_idx, valid_idx = indices[split:], indices[:split]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    return train_sampler, valid_sampler

def nonideal_nameAdd(model_name, en_nonideal):
    '''
    # this function generate model name with nonideal information
    
    # parameters:
    # model_name: model name(pregenerated)
    # en_nonideal: tuple of nonideal information, length unknown
    #              the sequence is: (SAF, thermal noise, shot noise, IR drop)
    '''
    
    nonideal_model_name = model_name
    # nonideal name list
    nonideal_name = ['SAF', 'ThermalNoise', 'ShotNoise', 'IRdrop']
    
    # condition check
    nonideal_index = [i for i, x in enumerate(en_nonideal) if x == 1]
        
    if len(nonideal_index) == 0:
        # no nonideal
        nonideal_model_name = nonideal_model_name
    elif len(nonideal_index) == 1:
        # 1 nonideal
        nonideal_model_name += '_' + nonideal_name[nonideal_index[0]]
    elif len(nonideal_index) == 2:
        # 2 nonideals
        nonideal_model_name += '_' + nonideal_name[nonideal_index[0]] + '_' + nonideal_name[nonideal_index[1]]
    elif len(nonideal_index) == 3:
        # 3 nonideals
        # find the one which doesnt contains
        nocontain_index = [i for i, x in enumerate(en_nonideal) if x == 0]
        nonideal_model_name += '_no' + nonideal_name[nocontain_index[0]]
    elif len(nonideal_index) == 4:
        # 4 nonideals
        nonideal_model_name += '_Nonideal'
    return nonideal_model_name

def model_result_save(model_name,
                      model_save_path,
                      train_loss,
                      val_loss,
                      test_loss,
                      val_acc,
                      test_acc,
                      lr):
    np.savetxt( model_save_path + '/' + model_name + '_train_loss.txt', train_loss, delimiter=',')
    np.savetxt( model_save_path + '/' + model_name + '_val_loss.txt', val_loss, delimiter=',')
    np.savetxt( model_save_path + '/' + model_name + '_test_loss.txt', test_loss, delimiter=',')
    np.savetxt( model_save_path + '/' + model_name + '_val_acc.txt', val_acc, delimiter=',')
    np.savetxt( model_save_path + '/' + model_name + '_test_acc.txt', test_acc, delimiter=',')
    np.savetxt( model_save_path + '/' + model_name + '_lr.txt', test_acc, delimiter=',')
    
    # loss plot
    plt.clf()
    plt.figure(1)
    plt.title(configs.get('DEFAULT', 'model')+' Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss, label = 'train loss', color = 'blue')
    plt.plot(val_loss, label = 'val loss', color = 'green')
    plt.plot(test_loss, label = 'test loss', color = 'orange')
    plt.legend()
    plt.savefig(model_save_path + '/' + model_name + '_loss.png')
    
    # acc plot
    plt.clf()
    plt.figure(2)
    plt.title(configs.get('DEFAULT', 'model')+' Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.plot(val_acc, label = 'val acc', color = 'green')
    plt.plot(test_acc, label = 'test acc', color = 'orange')
    plt.legend()
    plt.savefig(model_save_path + '/' + model_name + '_acc.png')
    
    # lr plot
    plt.clf()
    plt.figure(3)
    plt.title(configs.get('DEFAULT', 'model')+' Learning Rate')
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.plot(lr, color = 'purple')
    plt.savefig(model_save_path + '/' + model_name + '_lr.png')
