import torch
import random
import torch.nn.functional as F
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
## train: tv=1e-3  untrain: tv=1e-2
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

epoch=10
repetition=10
sample_per_class=3
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
    type='label_smooth',
    pretrained=bool(strtobool(args.pretrained)),
    lr=0.5,
    bound=100,
    iteration=200,
    initia=1.,
    coefficient=4)



dir_name='data/fc_recovery/ext/'+CONFIG['type']+'_'+str(args.tv)+'_cuda:'+args.cuda+'_'+cost_fn+'_'+str(CONFIG['pretrained'])+datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S')
test=label_recovery(CONFIG)
data_index_list=np.load("additional_files/mixup_list_cifar10.npy")
loss_fn_alex = LPIPS(net='alex')
loss_fn_vgg = LPIPS(net='vgg')


rec_label=np.zeros((epoch,sample_per_class,repetition,comparison,test.classes))

prob_list=np.zeros((epoch,sample_per_class))
psnr=np.zeros((epoch,sample_per_class,repetition,comparison))
ssim=np.zeros((epoch,sample_per_class,repetition,comparison))
lpips_alex=np.zeros((epoch,sample_per_class,repetition,comparison))
lpips_vgg=np.zeros((epoch,sample_per_class,repetition,comparison))
image_buffer=torch.zeros((epoch,sample_per_class,repetition,comparison,3,test.size[0],test.size[1]))
image_gt=torch.zeros((epoch,sample_per_class,3,test.size[0],test.size[1]))
runningloss=np.zeros((epoch,sample_per_class,repetition,comparison))
image_index=np.zeros((epoch,sample_per_class))
for i in range(epoch):
    image_index[i]=np.random.choice(data_index_list[i],3,replace=False)
if strtobool(args.verble):
    os.makedirs(dir_name)
    np.save(dir_name+"/image_index.npy",image_index)

for i in range(epoch):
    print(f"epoch={i}")
    for ii in range(sample_per_class):
        print(f"sample {ii}!")
        if hasattr(test,"recover_label"):
            del test.recover_label 
        while not hasattr(test,"recover_label"):
            prob=random.uniform(0,0.5)
            test.setup(int(image_index[i,ii]),prob)
            image_gt[i,ii]=test.origin_data[0].cpu()
            test.label_reco()
            if not hasattr(test,"recover_label"):
                test.pso()
            prob_list[i,ii]=prob
        if strtobool(args.verble):
            np.save(dir_name+'/prob_list.npy',prob_list)
            torch.save(image_gt,dir_name+"/image_gt.pt")
        for time in range(repetition):    
            print(f'repetition {time}!')   
            test.dummy_image=torch.randn(test.origin_data.size())
            test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True)
            image_buffer[i,ii,time,0]=test.dummy_data.detach().cpu()[0]
            psnr[i,ii,time,0]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
            ssim[i,ii,time,0]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
            lpips_alex[i,ii,time,0]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            lpips_vgg[i,ii,time,0]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            runningloss[i,ii,time,0]=test.runningloss
            rec_label[i,ii,time,0]=np.array(test.dummy_label.detach().cpu())

            test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True,method='f')
            image_buffer[i,ii,time,1]=test.dummy_data.detach().cpu()[0]
            psnr[i,ii,time,1]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
            ssim[i,ii,time,1]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
            lpips_alex[i,ii,time,1]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            lpips_vgg[i,ii,time,1]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            runningloss[i,ii,time,1]=test.runningloss
            rec_label[i,ii,time,1]=np.array(test.dummy_label.detach().cpu())

            test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True,method='gf')
            image_buffer[i,ii,time,2]=test.dummy_data.detach().cpu()[0]
            psnr[i,ii,time,2]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
            ssim[i,ii,time,2]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
            lpips_alex[i,ii,time,2]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            lpips_vgg[i,ii,time,2]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            runningloss[i,ii,time,2]=test.runningloss
            rec_label[i,ii,time,2]=np.array(test.dummy_label.detach().cpu())

            test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True,method='f',f_scalar=2)
            rec_label[i,ii,time,3]=np.array(test.dummy_label.detach().cpu())
            image_buffer[i,ii,time,3]=test.dummy_data.detach().cpu()[0]
            psnr[i,ii,time,3]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
            ssim[i,ii,time,3]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
            lpips_alex[i,ii,time,3]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            lpips_vgg[i,ii,time,3]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            runningloss[i,ii,time,3]=test.runningloss

            test.reconstruct(iteration=iteration, cost_fn=cost_fn, lr=lr, optim_fn=optim_fn, magnify=1,label='optimal',verble=verble,lr_decay=lr_decay,total_variation=total_v,keep=False,record_picking=True,method='gf',f_scalar=2)
            rec_label[i,ii,time,4]=np.array(test.dummy_label.detach().cpu())
            image_buffer[i,ii,time,4]=test.dummy_data.detach().cpu()[0]
            psnr[i,ii,time,4]=PSNR(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),data_range=256)
            ssim[i,ii,time,4]=SSIM(np.array(test.tp(test.origin_data[0].cpu())), np.array(test.tp(test.dummy_data[0].cpu())),channel_axis=2)
            lpips_alex[i,ii,time,4]=loss_fn_alex.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            lpips_vgg[i,ii,time,4]=loss_fn_vgg.forward(to_tensor(test.tp(test.origin_data[0].cpu())),to_tensor(test.tp(test.dummy_data[0].cpu())))
            runningloss[i,ii,time,4]=test.runningloss
            if strtobool(args.verble):
                torch.save(image_buffer,dir_name+"/image_buffer.pt")
                np.save(dir_name+"/psnr.npy",psnr)
                np.save(dir_name+"/ssim.npy",ssim)
                np.save(dir_name+"/lpips_alex.npy",lpips_alex)
                np.save(dir_name+"/lpips_vgg.npy",lpips_vgg)
                np.save(dir_name+"/runningloss.npy",runningloss)
                np.save(dir_name+"/rec_label.npy",rec_label)