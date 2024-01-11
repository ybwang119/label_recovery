import torch
import random
import torch.nn.functional as F
from itertools import combinations
from typing import OrderedDict
from recovering import label_recovery
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import os
from distutils.util import strtobool
from datetime import datetime
from lpips import LPIPS,im2tensor
import argparse
parser = argparse.ArgumentParser(description='setting for image recovery')
parser.add_argument('--seed',default=2023,type=int)
parser.add_argument('--pretrained',default='False',type=str)
parser.add_argument('--costfn',default='sim')
# parser.add_argument('--augmentation',default='label_smooth')
parser.add_argument('--verble',default='false',type=str)
parser.add_argument('--cuda',default='0',type=str)
parser.add_argument('--tv',default=1e-2,type=float)
args = parser.parse_args()
seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False  
torch.backends.cudnn.deterministic = True

epoch=45
repetition=10
comparison=5
cost_fn=args.costfn

def to_tensor(image):
    return im2tensor(np.array(image))

if cost_fn=='l2':
    iteration=300
    lr=1
    optim_fn='lbfgs'
    verble=50
    lr_decay=True
    total_v=args.tv
elif cost_fn=='sim':
    iteration=4800
    lr=0.1
    optim_fn='adam'
    verble=1000
    lr_decay=True
    total_v=args.tv


CONFIG=OrderedDict(device=torch.device('cuda:'+args.cuda),
    dataset="cifar10",
    network="lenet",
    opt="lbfgs",
    type='mixup',
    pretrained=bool(strtobool(args.pretrained)),
    lr=0.5,
    bound=100,
    iteration=200,
    initia=1.,
    coefficient=4)

dir_name='data/fc_recovery/ext/'+CONFIG['type']+'_'+str(args.tv)+'_cuda:'+args.cuda+'_'+cost_fn+'_'+str(CONFIG['pretrained'])+datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S')
test=label_recovery(CONFIG)
mixup_list=np.load("data/fc_recovery/mixup_list.npy")
loss_fn_alex = LPIPS(net='alex')
loss_fn_vgg = LPIPS(net='vgg')


rec_label=np.zeros((epoch,repetition,comparison,test.classes))

prob_list=np.zeros(epoch)
psnr=np.zeros((epoch,repetition,comparison))
ssim=np.zeros((epoch,repetition,comparison))
image_buffer=torch.zeros((epoch,repetition,comparison,3,test.size[0],test.size[1]))
image_gt=torch.zeros((epoch,3,3,test.size[0],test.size[1]))
runningloss=np.zeros((epoch,repetition,comparison))
image_index=np.zeros((epoch,2))
combination_list=list(combinations(range(10),2))
lpips_alex=np.zeros((epoch,repetition,comparison))
lpips_vgg=np.zeros((epoch,repetition,comparison))
if strtobool(args.verble):
    os.makedirs(dir_name)
    

for i in range(epoch):
    item=combination_list[i]
    print(f"epoch={i}")
    
    if hasattr(test,"recover_label"):
        del test.recover_label  
    while not hasattr(test,"recover_label"):
        ind=[random.randint(0,999),random.randint(0,999)]
        prob=np.random.beta(1,1)
        print(prob)
        image_index[i]=np.asarray([mixup_list[item[0],ind[0]],mixup_list[item[1],ind[1]]])
        test.setup([mixup_list[item[0],ind[0]],mixup_list[item[1],ind[1]]],[1-prob,prob])
        image_gt[i,2]=test.origin_data[0].cpu()
        image_gt[i,0]=test.mixup_buffer[0]
        image_gt[i,1]=test.mixup_buffer[1]
        test.label_reco()
        if not hasattr(test,"recover_label"):
            test.pso()
    prob_list[i]=prob
    if strtobool(args.verble):
        np.save(dir_name+'/prob_list.npy',prob_list)
        torch.save(image_gt,dir_name+"/image_gt.pt")
        np.save(dir_name+"/image_index.npy",image_index)
    for time in range(repetition):    
        print(f'repetition {time}!')   
        test.dummy_image=torch.randn(test.origin_data.size())
        test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True)
        image_buffer[i,time,0]=test.dummy_data.detach().cpu()[0]
        psnr[i,time,0]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
        ssim[i,time,0]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
        lpips_alex[i,time,0]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        lpips_vgg[i,time,0]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        runningloss[i,time,0]=test.runningloss
        rec_label[i,time,0]=np.array(test.dummy_label.detach().cpu())

        test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True,method='f')
        image_buffer[i,time,1]=test.dummy_data.detach().cpu()[0]
        psnr[i,time,1]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
        ssim[i,time,1]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
        lpips_alex[i,time,1]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        lpips_vgg[i,time,1]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        runningloss[i,time,1]=test.runningloss
        rec_label[i,time,1]=np.array(test.dummy_label.detach().cpu())

        test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True,method='gf')
        image_buffer[i,time,2]=test.dummy_data.detach().cpu()[0]
        psnr[i,time,2]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
        ssim[i,time,2]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
        lpips_alex[i,time,2]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        lpips_vgg[i,time,2]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        runningloss[i,time,2]=test.runningloss
        rec_label[i,time,2]=np.array(test.dummy_label.detach().cpu())

        test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True,method='f',f_scalar=2)
        rec_label[i,time,3]=np.array(test.dummy_label.detach().cpu())
        image_buffer[i,time,3]=test.dummy_data.detach().cpu()[0]
        psnr[i,time,3]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
        ssim[i,time,3]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
        lpips_alex[i,time,3]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        lpips_vgg[i,time,3]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        runningloss[i,time,3]=test.runningloss

        test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True,method='gf',f_scalar=2)
        rec_label[i,time,4]=np.array(test.dummy_label.detach().cpu())
        image_buffer[i,time,4]=test.dummy_data.detach().cpu()[0]
        psnr[i,time,4]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
        ssim[i,time,4]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
        lpips_alex[i,time,4]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        lpips_vgg[i,time,4]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
        runningloss[i,time,4]=test.runningloss
        if strtobool(args.verble):
            torch.save(image_buffer,dir_name+"/image_buffer.pt")
            np.save(dir_name+"/psnr.npy",psnr)
            np.save(dir_name+"/ssim.npy",ssim)
            np.save(dir_name+"/lpips_alex.npy",lpips_alex)
            np.save(dir_name+"/lpips_vgg.npy",lpips_vgg)
            np.save(dir_name+"/runningloss.npy",runningloss)
            np.save(dir_name+"/rec_label.npy",rec_label)