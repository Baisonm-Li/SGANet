import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
import os
import logging
from models.SGANet import SGANet


import logging
from data import H5Dataset,NPZDataset
import numpy as np
from utils import Metric,get_model_size,test_speed,beijing_time, set_logger,init_weights,set_seed
import argparse


parse = argparse.ArgumentParser()
parse.add_argument('--log_out',type=int,default=1)
parse.add_argument('--model_name',type=str)
parse.add_argument('--dataset',type=str)
parse.add_argument('--check_point',type=str,default=None)
parse.add_argument('--check_step',type=int,default=50)
parse.add_argument('--lr',type=int,default=4e-4)
parse.add_argument('--batch_size',type=int,default=32)
parse.add_argument('--epochs',type=int,default=1000)
parse.add_argument('--seed',type=int,default=3407)
args = parse.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = args.model_name
model = None
HSI_bands = 31
test_dataset_path = None
train_dataset_path = None

if args.dataset == 'CAVE':
    test_dataset_path = './datasets/CAVE_test.npz'
    train_dataset_path = './datasets/CAVE_test.npz'
    HSI_bands = 31
if args.dataset == "PaviaU":
    test_dataset_path = './datasets/PaviaU_test.npz'
    train_dataset_path = './datasets/PaviaU_train.npz'
    HSI_bands = 103
if args.dataset == "Havard":
    test_dataset_path = './datasets/Havard_test.npz'
    train_dataset_path = './datasets/Havard_test.npz'
    HSI_bands = 31
if args.dataset == "Urban":
    test_dataset_path = './datasets/Urban_test.npz'
    train_dataset_path = './datasets/Urban_test.npz'
    HSI_bands = 162

if model_name.startswith('SGANet'):
    model = SGANet(HSI_bands,hidden_dim=64)
    
model = model.to(device)
set_seed(args.seed)
loss_func = torch.nn.L1Loss()
optimizer = torch.optim.Adam(lr=args.lr,params=model.parameters())
scheduler = StepLR(optimizer=optimizer,step_size=100,gamma=0.1)
test_dataset = NPZDataset(test_dataset_path)
train_dataset = NPZDataset(train_dataset_path)
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size * 4)
start_epoch = 0

if args.check_point is not None:
    checkpoint = torch.load(args.check_point)  
    model.load_state_dict(checkpoint['net'],strict=False)  
    optimizer.load_state_dict(checkpoint['optimizer']) 
    start_epoch = checkpoint['epoch']+1 
    scheduler.load_state_dict(checkpoint['scheduler'])
    log_dir,_ = os.path.split(args.check_point)
    print(f'check_point: {args.check_point}')
    
if args.check_point is  None:
    init_weights(model)
    log_dir = f'./trained_models/{model_name},{args.dataset},{beijing_time()}'
    if not os.path.exists(log_dir) and args.log_out == 1:
        os.mkdir(log_dir)

logger = set_logger(model_name, log_dir, args.log_out)
model_size = get_model_size(model)
inference_time,flops,params = test_speed(model,device,HSI_bands)
logger.info(f'[model:{args.model_name},dataset:{args.dataset}],model_size:{params},inference_time:{inference_time:.6f}S,FLOPs:{flops}')

def train():
    model.train()
    loss_list = []
    for epoch in range(start_epoch, args.epochs):
        for idx,loader_data in enumerate(train_dataloader):
            GT,LRHSI,RGB = loader_data[0].to(device),loader_data[1].to(device),loader_data[2].to(device)
            optimizer.zero_grad()
            preHSI = model(LRHSI,RGB)
            loss = loss_func(GT,preHSI)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        test(epoch=epoch)

@torch.no_grad()
def test(epoch=-1):
    model.eval()
    loss_list, PSNR_list, SSIM_list, ERGAS_list, SAM_list = [],[],[],[],[]
    for idx,loader_data in enumerate(test_dataloader):
        GT,LRHSI,RGB = loader_data[0].to(device),loader_data[1].to(device),loader_data[2].to(device)
        preHSI = model(LRHSI,RGB)
        metric = Metric(GT,preHSI)
        loss = loss_func(GT,preHSI)
        loss_list.append(loss.item())
        PSNR_list.append(metric.PSNR)
        SSIM_list.append(metric.SSIM)
        ERGAS_list.append(metric.ERGAS)
        SAM_list.append(metric.SAM)
    if args.log_out == 1 and (epoch + 1) % args.check_step == 0: 
        checkpoint= {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'scheduler':scheduler.state_dict()
            }
        torch.save(checkpoint,f'{log_dir}/epoch:{epoch},PSNR:{np.mean(PSNR_list):.4f}.pth')
    logger.info(f'[Test:{model_name},{args.dataset}] epoch:{epoch}, loss:{np.mean(loss_list):.4f}, PSNR:{np.mean(PSNR_list):.4f}, SSIM:{np.mean(SSIM_list):.4f}, SAM:{np.mean(SAM_list):.4f}, ERGAS:{np.mean(ERGAS_list):.4f}')

if __name__ == "__main__":
    train()
