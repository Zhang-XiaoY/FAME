
import os
import datetime
import configparser as cp

# torch function
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.datasets as datasets
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms

# import self-defined net structure
from net_struct.LeNet_mnist import LeNet_MNIST

# import optimization method
from training_method.simple_train import simple_train

# import nonidealities
from weight_rebuild.smr import smr

# config reading
configs = cp.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config.ini')
configs.read(config_path)

cxb_sahpe = tuple(map(int,configs.get('DEFAULT', 'cxb_shape').split(',')))
min_g = configs.getfloat('DEFAULT', 'min_g')
max_g = configs.getfloat('DEFAULT', 'max_g')
wire_g = configs.getfloat('DEFAULT', 'wire_g')
SA0_ratio = configs.getfloat('DEFAULT', 'SA0_ratio')
SA1_ratio = configs.getfloat('DEFAULT', 'SA1_ratio')
en_nonidealities = [configs.getint('DEFAULT', 'en_SAF'),
                    configs.getint('DEFAULT', 'en_ThermalNoise'),
                    configs.getint('DEFAULT', 'en_ShotNoise'),
                    configs.getint('DEFAULT', 'en_rtn'),
                    configs.getint('DEFAULT', 'en_prog_error'),]

# data and model save setting
data_root_path = configs.get('DEFAULT', 'data_root_path') + '/data'
model_save_path = configs.get('DEFAULT', 'model_save_path') \
                  + '/' +datetime.datetime.now().strftime('%Y%m%d') \
                  + '_' + configs.get('DEFAULT', 'model')
model_name = 'ori_' + configs.get('DEFAULT', 'model')
dataset_name = configs.get('DEFAULT', 'dataset')
batch_size = configs.getint('DEFAULT', 'batch_size')


def main():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

    dataset = getattr(datasets, dataset_name)
    train_set = dataset(root= data_root_path, train=True,
                        transform=transform_train, download=True)
    test_set = dataset(root= data_root_path, train=False,
                    transform=transform_test, download=True,)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = LeNet_MNIST().to(device)
    
    model = smr(model, cxb_sahpe, min_g, max_g, wire_g, SA0_ratio, SA1_ratio, device, en_nonidealities)

    optimizer = optim.SGD(model.parameters(), 
                      lr=configs.getfloat('DEFAULT','learning_rate'), 
                      momentum=configs.getfloat('DEFAULT','momentum'), 
                      weight_decay=configs.getfloat('DEFAULT','weight_decay'))
    scheduler = lr_scheduler.ConstantLR(optimizer, 0.5, 10)

    simple_train(model,
             optimizer,
             scheduler,
             train_set=train_set,
             test_loader=test_loader,
             epoch_num=configs.getint('DEFAULT', 'epoch_num'))

if __name__ == '__main__':
    main()
