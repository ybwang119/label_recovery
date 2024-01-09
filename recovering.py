import torch
from torch import nn,optim
from copy import deepcopy
import pandas as pd
import torch.nn.functional as F
from metrics import total_variation as TV
from typing import OrderedDict
from PIL import Image
import time
from sko.PSO import PSO
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
from torch.optim import lr_scheduler
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from model import AlexNet, Bottleneck, BasicBlock,fc, ResNet,LeNet
print(torch.__version__, torchvision.__version__)
from resnet_label_smooth import ResNet18_ls
from resnet_mixup import ResNet18_mx
class label_recovery():
    
    def __init__(self,CONFIG) -> None:
        self.config=CONFIG
        self.tp=transforms.Compose([
                transforms.Normalize((-0.4914672374725342/0.24703224003314972, -0.4822617471218109/0.24348513782024384, -0.4467701315879822/0.26158785820007324), (1/0.24703224003314972, 1/0.24348513782024384, 1/0.26158785820007324)),
                transforms.ToPILImage()])
        self.tt=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822), (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])
        self.device=self.config['device']
        self.scalar=torch.tensor(self.config['initia']).to(self.device)
        self.criterion=self._cross_entropy_for_onehot#
        if self.config['dataset']=="flowers":
            self.size=(224,224)
            self.classes=17
            self.datadir="data/17flowers/"
            self.datalist="data/17flowers/dataset_flower.csv"
        elif self.config['dataset']=="imagenet":
            self.classes=1000
            self.size=(224,224)
            self.datadir="data/imagenet/pure__data/"
            self.datalist='data/imagenet/pure__data/validation_ground_truth.csv'   
        elif self.config['dataset']=="cifar10":
            self.classes=10
            self.size=(32,32)
            self.datadir="data/CIFAR10/test/"
            self.datalist="data/CIFAR10/test_ground_truth.csv"
        elif self.config['dataset']=="cifar100":
            self.size=(32,32)
            self.classes=100
            self.datadir="data/CIFAR100/test/"
            self.datalist="data/CIFAR100/test_ground_truth.csv" 
        
        if self.config['network']=="fc":
            self.net=fc(self.classes).to(self.device)
            self.forward_function=self.net.fc4
            self.flatten=True
        elif self.config['network']=="resnet50":
            self.net=ResNet(Bottleneck,[3,4,6,3],self.classes,False)
            if self.config['pretrained']:
                net_weight=torch.load("data/fc_recovery/resnet50-0676ba61.pth")
                self.net.load_state_dict(net_weight,False)
            else:
                self._weights_init()
            self.net.to(self.device)
            self.forward_function=self.net.linear
            self.flatten=False
        elif self.config['network']=="resnet18":
            #self.net=torchvision.models.resnet18(False)
            if self.config['pretrained']:
                if self.config['type']=='label_smooth' and self.config['dataset']=='cifar10':
                    self.net=ResNet18_ls(10)
                    net_weight=torch.load('/home/yanbo.wang/data/label_smooth_checkpoint/2023-04-07 12:56:01resnet_modified.pth',map_location=self.device)['net']
                    self.forward_function=self.net.linear
                elif self.config['type']=='mixup'and self.config['dataset']=='cifar10':
                    net_weight=torch.load('/home/yanbo.wang/data/mixup_checkpoint/2023-04-10 15:33:47 ResNet_95.45.pth',map_location=self.device)
                    self.net=ResNet18_mx(10)
                    self.forward_function=self.net.linear
                elif self.config['type']=='label_smooth'and self.config['dataset']=='flowers':
                    net_weight=torch.load('/home/yanbo.wang/data/label_smooth_checkpoint/2023-05-05 19:13:31flower_resnet_18_83.611.pth',map_location=self.device)['net']
                    self.net=ResNet18_ls(17)
                    self.forward_function=self.net.linear
                elif self.config['type']=='mixup'and self.config['dataset']=='flowers':
                    net_weight=torch.load('/home/yanbo.wang/data/mixup_checkpoint/2023-05-05 19:31:08 resnet_flowers_pa.pth',map_location=self.device)
                    self.net=ResNet18_mx(17)
                    self.forward_function=self.net.linear
                self.net.load_state_dict(net_weight)
            else:
                self.net=ResNet(BasicBlock, [2,2,2,2],self.classes,False)
                self.forward_function=self.net.linear
            self.net.to(self.device)
            
            self.flatten=False
        elif self.config['network']=="lenet":
            self.net=LeNet(self.classes)
            if self.config['pretrained']:
                if self.config['type']=='label_smooth' and self.config['dataset']=='cifar10':
                    net_weight=torch.load('/home/yanbo.wang/data/label_smooth_checkpoint/2023-04-07 13:48:55lenet_modified.pth',map_location=self.device)['net']
                    self.net.load_state_dict(net_weight)

                elif self.config['type']=='mixup' and self.config['dataset']=='cifar10':
                    net_weight=torch.load('/home/yanbo.wang/data/mixup_checkpoint/lenet59.84.pth',map_location=self.device)
                    self.net.load_state_dict(net_weight)
            else:
                self._weights_init()
            self.net.to(self.device)
            self.forward_function=self.net.fc
            self.flatten=False
        
        elif self.config['network']=="alexnet":
            self.net=AlexNet(self.classes)
            if self.config['pretrained']:
                if self.config['type']=='label_smooth' and self.config['dataset']=='flowers':
                    net_weight=torch.load('/home/yanbo.wang/data/label_smooth_checkpoint/2023-05-05 19:09:42flower_alexnet_99.pth',map_location=self.device)['net']
                    self.net.load_state_dict(net_weight)

                elif self.config['type']=='mixup' and self.config['dataset']=='flowers':
                    net_weight=torch.load('/home/yanbo.wang/data/mixup_checkpoint/2023-05-05 20:50:07 alexnet_flowers_pa.pth',map_location=self.device)
                    self.net.load_state_dict(net_weight)

            else:
                self._weights_init()
            self.net.to(self.device)
            self.forward_function=self.net.fc
            self.flatten=False
        self.net.eval()
    def _weights_init(self):
        for i in self.net.modules():
            if hasattr(i, "weight"):
                i.weight.data.uniform_(-0.5, 0.5)
            if hasattr(i, "bias") and hasattr(i.bias, "data"):
                i.bias.data.uniform_(-0.5, 0.5)

    def _label_to_onehot(self, target, num_classes):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def _cross_entropy_for_onehot(self, pred, target):
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    
    
    
    def _mixup(self,ind,lamda:list):
        self.mixup_buffer=[]
        data=pd.read_csv(self.datalist,header=None)
        for index,i in enumerate(ind):
            name=self.datadir+data.iloc[i][0]
            image=Image.open(name).convert("RGB").resize(self.size)
            self.mixup_buffer.append(self.tt(image))
            if index==0:
                mixup_image=self.tt(image)*lamda[0]
                mixup_label=self._label_to_onehot(torch.tensor(data.iloc[i][1],dtype=torch.int64).unsqueeze(0),self.classes)*lamda[0]
            else:
                mixup_image+=self.tt(image)*lamda[index]
                mixup_label+=self._label_to_onehot(torch.tensor(data.iloc[i][1],dtype=torch.int64).unsqueeze(0),self.classes)*lamda[index]
        if self.flatten:
            mixup_image=torch.flatten(mixup_image,0,-1)
        index=lamda.index(max(lamda))
        hard_label=self._label_to_onehot(torch.tensor(data.iloc[ind[index]][1],dtype=torch.int64).unsqueeze(0),self.classes)
        return mixup_image.unsqueeze(0),mixup_label,hard_label

    def _label_smooth(self,ind,lamda):
        data=pd.read_csv(self.datalist,header=None)
        name=self.datadir+data.iloc[ind][0]
        image=self.tt(Image.open(name).convert("RGB").resize(self.size))
        if self.flatten:
            image=torch.flatten(image,0,-1)
        hard_label=self._label_to_onehot(torch.tensor(data.iloc[ind][1],dtype=torch.int64).unsqueeze(0),self.classes)
        smoothed_label=deepcopy(hard_label)
        for index,i in enumerate(smoothed_label[0]):
            if i==0:
                smoothed_label[0][index]=i+lamda/smoothed_label.size(1)
            else:
                smoothed_label[0][index]=i-lamda+lamda/smoothed_label.size(1)
        return image.unsqueeze(0),smoothed_label,hard_label

    def setup(self,ind,lamda,type="variant",noise=None,std=1e-4):
        self.dataset_total=pd.read_csv(self.datalist,header=None).shape[0]

        if self.config['type']=='label_smooth':
            self.origin_data,self.origin_label,self.hard_label=self._label_smooth(ind=ind,lamda=lamda)

        elif self.config['type']=='mixup':
            self.origin_data,self.origin_label,self.hard_label=self._mixup(ind=ind,lamda=lamda)
        self.origin_data,self.origin_label=self.origin_data.to(self.device),self.origin_label.to(self.device)
        self.hard_label=self.hard_label.to(self.device)

        y=self.net(self.origin_data)
        if type=="hard_label":
            loss=self.criterion(y,self.hard_label)
        elif type=='variant':
            #测试稳定性，hardlabel的梯度跟之前进行比较
            loss=self.criterion(y,self.origin_label)
        self.net.zero_grad()
        dy_dx = torch.autograd.grad(loss, self.net.parameters(),retain_graph=True)
        #self.dy_dx=dy_dx
        self.whole_gradient = list((_.detach().clone() for _ in dy_dx))
        self.net.zero_grad()
        loss.backward()
        self.input_ground_truth=self.net.temp
        self.gradient=deepcopy(self.forward_function.weight.grad)
        init_pick=torch.argmax(abs(self.forward_function.weight.grad.sum(dim=1)))
        recover_init=deepcopy(self.forward_function.weight.grad[init_pick])# refer to x
        self.recover_init=recover_init/self.config['coefficient']
        self.ground_truth=np.nanmean((self.net.temp/self.recover_init).cpu().detach())
        if noise!= None:
            if noise=='gaussian':
                #print(self.recover_init.size())
                self.noise=torch.normal(mean=0,std=std,size=self.recover_init.size())
                #print(recover_init)
                #print(self.noise)
                self.recover_init=(recover_init+self.noise)/self.config['coefficient']
                #print(self.recover_init)
            elif noise=='laplace':
                self.noise=np.random.laplace(0,scale=std/1.414,size=self.recover_init.size())
                self.noise=torch.from_numpy(self.noise).float()
                self.recover_init=(recover_init+self.noise)/self.config['coefficient']
        self.recover_init.requires_grad_(False)
        self.dummy_image=torch.randn(self.origin_data.size())
        return

    def _fake_y(self,input):
        prob=nn.functional.softmax(self.forward_function(input),dim=-1)
        #print(prob)
        if max(input)!=0:
            arg=torch.argmax(input) 
        else:
            arg=torch.argmin(input)
        y=torch.zeros(self.gradient.shape[0]).to(self.device)
        for index,i in enumerate(self.gradient):
            y[index]=prob[index]-i[arg]/input[arg]
        return y,prob
    def _pick_closure(self):
        if self.config['type']=="label_smooth":
            def closure():
                self.optimizer.zero_grad()
                presdo_y,_=self._fake_y(self.scalar*self.recover_init)
                top=torch.topk(presdo_y,self.classes-1,largest=False,sorted=False).values
                loss=torch.var(top,unbiased=False)*1000
                loss.backward(retain_graph=True)
                return loss 
        elif self.config['type']=="mixup":
            def closure():
                self.optimizer.zero_grad()
                presdo_y,_=self._fake_y(self.scalar*self.recover_init)
                top=torch.topk(presdo_y,self.classes-2,largest=False,sorted=False).values
                loss=torch.var(top,unbiased=False)*1000
                loss.backward(retain_graph=True)
                return loss
        return closure  
    def label_reco(self):
        closure=self._pick_closure()
        # if math.isinf(self.ground_truth):
        #     return
        if torch.equal(self.gradient,torch.zeros_like(self.gradient)):
            print("here!!!!!!!")
            return 0
        flip=False
        if self.config['opt']=="lbfgs" or "lgfbs":
            self.scalar=torch.tensor(self.config['initia']).to(self.device)# refer to lamda
            self.scalar.requires_grad=True
            self.optimizer=optim.LBFGS([self.scalar],lr=self.config['lr'])
            buffer=[0,0,0]
            skip_weight=0.5
            for epoch in range(self.config['iteration']//2):
                
                loss=closure()
                #print(loss.item())
                buffer.pop(0)
                buffer.append(loss.item())
                if abs(self.scalar)>self.config['bound']:
                    temp_loss=loss.data
                    temp_scalar=self.scalar.data
                    print("flip!")
                    temp_index=epoch
                    break
                elif loss<1e-9:
                    if abs(self.scalar.item()-self.ground_truth.item())<1e-2:
                        self.recover_tensor=self.scalar*self.recover_init
                        self.recover_tensor.detach_()
                        self.recover_label,_=self._fake_y(self.recover_tensor)
                        # print(f"gradient tensors are {self.recover_init}")
                        # print(f"logits are {self.forward_function(self.scalar*self.recover_init)}")
                        print("epoch is "+str(epoch+1))
                        flip=True
                        return loss.item()
                    else:
                        print("scalar is "+str(self.scalar)+" while gt is "+str(self.ground_truth))
                        print(f"probability is {self._fake_y(self.scalar*self.recover_init)}")
                        print(f"ground-truth probability is {self._fake_y(self.ground_truth*self.recover_init)}")
                        print(f"gradient tensors are {self.recover_init}")
                        print(f"logits are {self.forward_function(self.scalar*self.recover_init)}")
                        print(f"layer weight is {self.forward_function.weight}")
                        print(f"input ground-truth feature is {self.input_ground_truth}")
                        print(f"input init is {self.recover_init}")
                        print("fail to find the ground-truth scalar!")
                        return -loss.item()
                elif max(buffer[0],buffer[1],buffer[2])-min(buffer[0],buffer[1],buffer[2])<(buffer[0]+buffer[1]+buffer[2])/3000:
                #elif buffer[1]==0:
                    print("skip!")
                    self.scalar.requires_grad_(False)
                    self.scalar+=skip_weight
                    skip_weight*=2
                    self.scalar.requires_grad_(True)
                    continue
                self.optimizer.step(closure)
                if epoch==self.config['iteration']//2-1:
                    temp_loss=loss.data
                    temp_scalar=self.scalar.data
                    temp_index=epoch
            if flip==False:
                buffer=[0,0,0]
                skip_weight=0.5
                self.scalar=torch.tensor(-self.config['initia'],requires_grad=True)# refer to lamda
                self.optimizer=optim.LBFGS([self.scalar],lr=self.config['lr'])
                for epoch in range(self.config['iteration']//2,self.config['iteration']):
                    loss=closure()
                    buffer.pop(0)
                    buffer.append(loss.item())
                    #print(loss)
                    #print(scalar)
                    if abs(self.scalar)>self.config['bound']:
                        print("scalar is "+str(self.scalar)+" while gt is "+str(self.ground_truth))
                        print("out of bound!")
                        return -1
                    elif loss<1e-9:
                        if abs(self.scalar.item()-self.ground_truth.item())<1e-2:
                            self.recover_tensor=self.scalar*self.recover_init
                            self.recover_tensor.detach_()
                            self.recover_label,_=self._fake_y(self.recover_tensor)
                            print("epoch is "+str(epoch-self.config['iteration']//2+temp_index+2))
                            return loss.item()
                        else:
                            print("scalar is "+str(self.scalar)+" while gt is "+str(self.ground_truth))
                            print(self._fake_y(self.scalar*self.recover_init))
                            print("fail to find the ground-truth scalar!")
                            return -loss.item()
                    elif max(buffer[0],buffer[1],buffer[2])-min(buffer[0],buffer[1],buffer[2])<(buffer[0]+buffer[1]+buffer[2])/3000:
                    #elif buffer[1]==0:
                        print("skip!")
                        self.scalar.requires_grad_(False)
                        self.scalar-=skip_weight
                        skip_weight*=2
                        self.scalar.requires_grad_(True)
                        continue
                    self.optimizer.step(closure)
                    if epoch==self.config['iteration']-1 and loss.data>temp_loss:
                        self.scalar.data=temp_scalar
                print("scalar is "+str(self.scalar)+" while gt is "+str(self.ground_truth))
                print(f"probability is {self._fake_y(self.scalar*self.recover_init)}")
                print(f"ground-truth probability is {self._fake_y(self.ground_truth*self.recover_init)}")
                print(f"gradient tensors are {self.recover_init}")
                print(f"logits are {self.forward_function(self.scalar*self.recover_init)}")
                print(f"layer weight is {self.forward_function.weight}")
                print(f"input ground-truth feature is {self.input_ground_truth}")
                print(f"input init is {self.recover_init}")
                print("unable to find the ground-truth scalar!")
                return -1
    def _image_reconstruct_loss(self,dummy_dy_dx,cost_fn,method,f_scalar):
        grad_diff=0
        avoid=False
        if 'g' not in method:
            avoid=True
        if cost_fn=='l2':
            for ind,(gx, gy) in enumerate(zip(dummy_dy_dx, self.whole_gradient)): 
                grad_diff += ((gx - gy).pow(2)).sum()
                if ind==len(self.whole_gradient)-2 and avoid:
                    break
        elif cost_fn=='local_sim':
            for ind,(gx, gy) in enumerate(zip(dummy_dy_dx, self.whole_gradient)): 
                grad_diff+=1-torch.nn.functional.cosine_similarity(gx.flatten(),gy.flatten(),0,1e-10)
                if ind==len(self.whole_gradient)-2 and avoid:
                    break
        elif cost_fn=="sim":
            pnorm=[0,0]
            costs=0
            for ind,(gx, gy) in enumerate(zip(dummy_dy_dx, self.whole_gradient)): 
                costs += (gx * gy).sum()
                pnorm[0] += gx.pow(2).sum()
                pnorm[1] += gy.pow(2).sum()
                if ind==len(self.whole_gradient)-2 and avoid:
                    break
            grad_diff=1-costs/pnorm[0].sqrt()/pnorm[1].sqrt()
        if 'f' in method:
            if cost_fn=='l2':
                feature_loss=f_scalar*((self.net.temp-self.recover_tensor)**2).sum()
                grad_diff+=feature_loss
            elif cost_fn=='sim' or 'sim_local':
                feature_loss=f_scalar-f_scalar*((self.net.temp*self.recover_tensor).sum()/self.net.temp.pow(2).sum().sqrt()/self.recover_tensor.pow(2).sum().sqrt())
                grad_diff+=feature_loss
        return grad_diff
    def pso(self,inter_bound=20,bound=70):
        if self.device !=torch.device('cpu'):
            self.recover_init=self.recover_init.to(torch.device('cpu'))
            self.net=self.net.to(torch.device('cpu'))
        if self.config['type']=="label_smooth":
            def fun(x):
                presdo_y,_=self._fake_y(torch.Tensor(x)*self.recover_init)
                top=torch.topk(presdo_y,self.classes-1,largest=False,sorted=False).values
                loss=torch.var(top,unbiased=False)*1000
                return loss.cpu().detach().numpy() 
        elif self.config['type']=="mixup":
            def fun(x):
                presdo_y,_=self._fake_y(torch.Tensor(x)*self.recover_init)
                top=torch.topk(presdo_y,self.classes-2,largest=False,sorted=False).values
                loss=torch.var(top,unbiased=False)*1000
                return loss.cpu().detach().numpy() 
 
        pso_pl=1.
        #inter=[1,3,5,10,20]
        inter=[5,inter_bound/5,]
        index=0
        pso_pu=inter[index]+pso_pl
        print(f"ground_truth: {self.ground_truth}")
        while(pso_pu<bound):
            pso = PSO(func=fun, n_dim=1,max_iter=40,pop=500,ub=[pso_pu],lb=[pso_pl-0.3],verbose=False)   
            print(f"searching from {pso_pl-0.3} to {pso_pu}!")

            pso.run()
            print(pso.best_x,pso.best_y)
            if pso.gbest_y<1e-9:
                if abs(pso.gbest_x-self.ground_truth)<0.01:
                    self.recover_init=self.recover_init.to(self.device)
                    self.net=self.net.to(self.device)
                    self.scalar=torch.tensor(pso.gbest_x,dtype=torch.float).to(self.device)
                    self.recover_tensor=self.scalar*self.recover_init
                    self.recover_label,_=self._fake_y(self.recover_tensor)
                    print(f"successfully find the ground_truth {pso.gbest_x}")
                    return 0
                else:
                    print(f"Fail to find the ground_truth {self.ground_truth}")
                    self.recover_init=self.recover_init.to(self.device)
                    self.net=self.net.to(self.device)
                    return -1
            pso = PSO(func=fun, n_dim=1,max_iter=40,pop=500,ub=[-pso_pl],lb=[-pso_pu-0.3],verbose=False)   
            print(f"searching from {-pso_pu-0.3} to {-pso_pl}!")
            pso.run()
            print(pso.best_x,pso.best_y)
            if pso.gbest_y<1e-9:
                if abs(pso.gbest_x-self.ground_truth)<0.01:
                    self.recover_init=self.recover_init.to(self.device)
                    self.net=self.net.to(self.device)
                    self.scalar=torch.tensor(pso.gbest_x,dtype=torch.float).to(self.device)
                    self.recover_tensor=self.scalar*self.recover_init
                    self.recover_label,_=self._fake_y(self.recover_tensor)
                    print(f"successfully find the ground_truth {pso.gbest_x}")
                    return 1
                else:
                    print(f"Fail to find the ground_truth {self.ground_truth}")
                    self.recover_init=self.recover_init.to(self.device)
                    self.net=self.net.to(self.device)
                    return -1
            pso_pl=pso_pu
            index+=1
            if index>=len(inter):
                pso_pu+=inter_bound
            else:
                pso_pu+=inter[index]
        print(f"out of bound! The ground_truth: {self.ground_truth}")
        self.recover_init=self.recover_init.to(self.device)
        self.net=self.net.to(self.device)
        return -2
        # print('best_x is ',pso.gbest_x)
        # print('best_y is ',pso.gbest_y)
        # print("ground_truth is "+str(self.ground_truth))
    def pso_noise(self,inter_bound=5,bound=10):
        if self.device !=torch.device('cpu'):
            self.recover_init=self.recover_init.to(torch.device('cpu'))
            self.net=self.net.to(torch.device('cpu'))
        if self.config['type']=="label_smooth":
            def fun(x):
                presdo_y,_=self._fake_y(torch.Tensor(x)*self.recover_init)
                top=torch.topk(presdo_y,self.classes-1,largest=False,sorted=False).values
                loss=torch.var(top,unbiased=False)*1000
                return loss.cpu().detach().numpy() 
        elif self.config['type']=="mixup":
            def fun(x):
                presdo_y,_=self._fake_y(torch.Tensor(x)*self.recover_init)
                top=torch.topk(presdo_y,self.classes-2,largest=False,sorted=False).values
                loss=torch.var(top,unbiased=False)*1000
                return loss.cpu().detach().numpy() 
 
        pso_pl=1.
        #inter=[1,3,5,10,20]
        inter=[4,]
        index=0
        bestx=[]
        besty=[]
        pso_pu=inter[index]+pso_pl
        print(f"ground_truth: {self.ground_truth}")
        while(pso_pu<=bound):
            pso = PSO(func=fun, n_dim=1,max_iter=30,pop=200,ub=[pso_pu],lb=[pso_pl-0.3],verbose=False)   
            print(f"searching from {pso_pl-0.3} to {pso_pu}!")
            
            pso.run()
            print(pso.best_x,pso.best_y)
            bestx.append(pso.best_x[0])
            besty.append(pso.best_y[0])
            pso = PSO(func=fun, n_dim=1,max_iter=30,pop=200,ub=[-pso_pl],lb=[-pso_pu-0.3],verbose=False)   
            print(f"searching from {-pso_pu-0.3} to {-pso_pl}!")
            pso.run()
            print(pso.best_x,pso.best_y)
            bestx.append(pso.best_x[0])
            besty.append(pso.best_y[0])
            pso_pl=pso_pu
            index+=1
            if index>=len(inter):
                pso_pu+=inter_bound
            else:
                pso_pu+=inter[index]
        pointer=besty.index(min(besty))
        if abs(bestx[pointer]-self.ground_truth)<1:
            self.recover_init=self.recover_init.to(self.device)
            self.net=self.net.to(self.device)
            self.scalar=torch.tensor(bestx[pointer],dtype=torch.float).to(self.device)
            self.recover_tensor=self.scalar*self.recover_init
            self.recover_label,_=self._fake_y(self.recover_tensor)
            return min(besty)
        else:
            self.recover_init=self.recover_init.to(self.device)
            self.net=self.net.to(self.device)
            self.scalar=torch.tensor(bestx[pointer],dtype=torch.float).to(self.device)
            self.recover_tensor=self.scalar*self.recover_init
            self.recover_label,_=self._fake_y(self.recover_tensor)
            return -min(besty)
    def create_opt_dlg_label(self):
        softmax=torch.nn.Softmax(dim=1)
        input=torch.rand_like(self.origin_label,requires_grad=True).to(self.device)
        while ((softmax(input)-self.origin_label)**2).sum()>1e-8:
            input=torch.rand_like(self.origin_label,requires_grad=True).to(self.device)
            optimizer=torch.optim.AdamW([input],lr=0.01)
            
            count=0
            while ((softmax(input)-self.origin_label)**2).sum()>1e-8 and count<35000:
                count+=1
                def closure():
                    optimizer.zero_grad()
                    output=softmax(input)
                    loss=((output-self.origin_label)**2).sum()
                    loss.backward()
                    return loss
                optimizer.step(closure)
        self.opt_dlg_label=input.detach()
    def reconstruct(self,iteration,cost_fn,lr,optim_fn,lr_decay=True,
    total_variation=1e-2,verble=1000,label='optimal',method="g",magnify=1,keep=False,record_picking=False,f_scalar=1):
        start_time = time.time()
        # buffer=[]
        
        if not keep:
            self.dummy_data=deepcopy(self.dummy_image.detach().to(self.device).requires_grad_(True))
            def loss_fn(pred, labels):
                post_label = torch.nn.functional.softmax(labels, dim=-1)
                return torch.mean(torch.sum(-post_label * torch.nn.functional.log_softmax(pred, dim=-1), 1))
            if label=='dlg':
                if record_picking:
                    self.buffer_label=[]
                self.dummy_label=torch.randn(self.origin_label.size()).to(self.device).requires_grad_(True)              
                self.match_criterion=loss_fn
                if optim_fn=="adam":
                    image_optimizer=torch.optim.Adam([self.dummy_data,self.dummy_label],lr=lr)
                elif optim_fn=="lbfgs" or "lgfbs":
                    image_optimizer = torch.optim.LBFGS([self.dummy_data,self.dummy_label],lr=lr)
            elif label=='opt_dlg':
                if record_picking:
                    self.buffer_label=[]
                self.match_criterion=loss_fn
                self.dummy_label=deepcopy(self.opt_dlg_label)
                self.dummy_label.to(self.device).requires_grad_(True)
                if optim_fn=="adam":
                    image_optimizer=torch.optim.Adam([self.dummy_data,self.dummy_label],lr=lr)
                elif optim_fn=="lbfgs" or "lgfbs":
                    image_optimizer = torch.optim.LBFGS([self.dummy_data,self.dummy_label],lr=lr)
            elif label=='optimal' or label=='origin' or label=="hard":
                self.match_criterion=self._cross_entropy_for_onehot
                if label=='optimal':
                    self.dummy_label=deepcopy(self.recover_label.detach())
                    self.dummy_label.to(self.device).requires_grad_(False)        
                elif label=='origin':
                    self.dummy_label=deepcopy(self.origin_label.detach())
                    self.dummy_label.to(self.device).requires_grad_(False)
                elif label=='hard':
                    self.dummy_label=deepcopy(self.hard_label.detach())
                    self.dummy_label.to(self.device).requires_grad_(False)
                if optim_fn=="adam":
                    image_optimizer = torch.optim.Adam([self.dummy_data],lr=lr)
                elif optim_fn=="lbfgs" or "lgfbs":
                    image_optimizer = torch.optim.LBFGS([self.dummy_data],lr=lr)
            if lr_decay:
                image_scheduler = lr_scheduler.MultiStepLR(image_optimizer,
                                                                milestones=[iteration // 2.667, iteration // 1.6,
                                                                            iteration // 1.142], gamma=0.1)   # 3/8 5/8 7/8     
            if record_picking:
                self.buffer_image=[self.dummy_data.cpu()]
                self.buffer_loss=[]
                if self.dummy_label.requires_grad:
                    self.buffer_label=[self.dummy_label.cpu()]
                if self.flatten:
                    self.PSNR=[PSNR(np.asarray(self.tp(self.origin_data[0].cpu().reshape((3,64,64)))), np.asarray(self.tp(self.dummy_data[0].cpu().reshape((3,64,64)))),data_range=256)]
                    self.SSIM=[SSIM(np.asarray(self.tp(self.origin_data[0].cpu().reshape((3,64,64)))), np.asarray(self.tp(self.dummy_data[0].cpu().reshape((3,64,64)))),channel_axis=2)]
                else:
                    self.PSNR=[PSNR(np.asarray(self.tp(self.origin_data[0].cpu())), np.asarray(self.tp(self.dummy_data[0].cpu())),data_range=256)]
                    self.SSIM=[SSIM(np.asarray(self.tp(self.origin_data[0].cpu())), np.asarray(self.tp(self.dummy_data[0].cpu())),channel_axis=2)]
        for iters in range(iteration):
            def closure():
                image_optimizer.zero_grad()
                self.net.zero_grad()
                dummy_pred = self.net(self.dummy_data) 
                dummy_loss = self.match_criterion(dummy_pred, self.dummy_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net.parameters(),create_graph=True)        
                grad_diff=self._image_reconstruct_loss(dummy_dy_dx,cost_fn,method=method,f_scalar=f_scalar)
                if total_variation>0:
                    if self.flatten==True:
                        grad_diff+=total_variation*TV(self.dummy_data.reshape((1,3,64,64)))
                    else:
                        grad_diff+=total_variation*TV(self.dummy_data)
                grad_diff*=magnify
                grad_diff.backward()
                return grad_diff   
            self.runningloss=image_optimizer.step(closure)
            self.runningloss=self.runningloss.item()
            if record_picking:
                self.buffer_image.append(self.dummy_data.cpu())
                self.buffer_loss.append(self.runningloss)
                if self.dummy_label.requires_grad:
                    self.buffer_label.append(self.dummy_label.cpu())
                if self.flatten:
                    self.PSNR.append(PSNR(np.asarray(self.tp(self.origin_data[0].cpu().reshape((3,64,64)))), np.asarray(self.tp(self.dummy_data[0].cpu().reshape((3,64,64)))),data_range=256))
                    self.SSIM.append(SSIM(np.asarray(self.tp(self.origin_data[0].cpu().reshape((3,64,64)))), np.asarray(self.tp(self.dummy_data[0].cpu().reshape((3,64,64)))),channel_axis=2))
                else:
                    self.PSNR.append(PSNR(np.asarray(self.tp(self.origin_data[0].cpu())), np.asarray(self.tp(self.dummy_data[0].cpu())),data_range=256))
                    self.SSIM.append(SSIM(np.asarray(self.tp(self.origin_data[0].cpu())), np.asarray(self.tp(self.dummy_data[0].cpu())),channel_axis=2))
            if lr_decay:
                image_scheduler.step()
            if iters==0 or iters%verble==verble-1:
                print(f"Iter: {iters+1}; Loss: {self.runningloss}")
        if record_picking:
            self.buffer_loss.append(closure().item())
            self.pick_best("loss")   
        print(f'Total time: {time.time()-start_time}.')
    def pick_best(self,choice="loss"):
        if choice=="loss":
            index=np.nanargmin(np.asarray(self.buffer_loss))
        elif choice=="psnr":
            index=np.nanargmax(np.asarray(self.PSNR))
        elif choice=="ssim":
            index=np.nanargmax(np.asarray(self.SSIM))
        self.dummy_data=self.buffer_image[index]
        self.runningloss=self.buffer_loss[index]
        if self.dummy_label.requires_grad:
            self.dummy_label=self.buffer_label[index]