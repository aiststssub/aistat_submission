import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from source.base import f, h, log_prob, log_prob_unnormalized
#np.set_printoptions(precision=3)
plt.style.use('seaborn-white')
torch.manual_seed(1234)
saveroot = 'trajectories'


mu_p = torch.tensor([0.0, 0.0])
scale_p = torch.tensor([[0.5, 0.3],
                        [0.3, 0.5]])

    

def train(detach=False, func='f', method='Rkl', number_iter=1000, lr=0.001):
    
    mu_q = torch.tensor([1.0, 0.5])
    scale_q = torch.tensor([[1.0000, 0.0000],
                        [0.0, 1.0]])

    mu_q.requires_grad = True
    scale_q.requires_grad = True
    opt = torch.optim.SGD([mu_q, scale_q], lr=lr)
    
    mus = np.zeros([number_iter, 2])
   
    for i in range(number_iter):
        mus[i] = mu_q.detach()
        opt.zero_grad()
        
        z = torch.randn([100, 2])
        
        # reparameterization trick
        x = mu_q + z @ scale_q.T
        if not detach:
            log_ratio = log_prob(mu_p, scale_p, x) - log_prob(mu_q, scale_q, x)
        else:
            log_ratio = log_prob_unnormalized(mu_p, scale_p, x) - log_prob(mu_q.detach(), scale_q.detach(), x)
            
        r = torch.exp(log_ratio)
        if func == 'f':
            loss = torch.mean(f(r, method))
        if func == 'h':
            loss = -torch.mean(h(r, method))  
        loss.backward()
        opt.step()

    return mus


def save_plot(mu1, mu2, name):

    filepath_fig1 = os.path.join(saveroot, f"small-{name}.png")
    plt.figure(figsize=(4,4))
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.scatter(x=mu_p[0], y=mu_p[1], s=40, label='Target mean', alpha=1.0, color='red', zorder=1)
    plt.scatter(x=mu1[0][0], y=mu1[0][1], s=40, label='Initial mean', alpha=1.0, color='orange', zorder=1)
    plt.plot(mu1[:, 0], mu1[:, 1], label='BBVI-path', alpha=1.0, color='blue', zorder=0)
    plt.plot(mu2[:, 0], mu2[:, 1], label='BBVI-rep', alpha=0.6, color='green', zorder=0)

    plt.ylim(-0.05, 0.7)
    #plt.axis('equal')
    plt.legend(prop={'size': 12}, loc='upper left')
    plt.savefig(filepath_fig1)
    plt.close() 
    
           
    filepath_fig2 = os.path.join(saveroot, f"big-{name}.png")
    plt.figure(figsize=(4,4))
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.scatter(x=mu_p[0], y=mu_p[1], s=40, label='Target mean', alpha=1.0, color='red', zorder=1)
    plt.plot(mu1[:, 0], mu1[:, 1], label='BBVI-path', alpha=1.0, color='blue', zorder=0)
    plt.plot(mu2[:, 0], mu2[:, 1], label='BBVI-rep', alpha=0.6, color='green', zorder=0)

    plt.ylim(-0.05, 0.1)
    plt.xlim(-0.05, 0.1)
    #plt.axis('equal')
    plt.legend(prop={'size': 12}, loc='upper left')
    plt.savefig(filepath_fig2)
    plt.close() 
    
    
    
if __name__ == '__main__':

    if not(os.path.isdir(saveroot)):
        os.mkdir(saveroot)
    mus_f_rkl = train(detach=False, func='f', method='Rkl', number_iter=1300, lr=0.003)
    mus_h_rkl = train(detach=True, func='h', method='Rkl', number_iter=1300, lr=0.003)
    
    mus_f_fkl = train(detach=False, func='f', method='Fkl', number_iter=2500, lr=0.003)
    mus_h_fkl = train(detach=True, func='h', method='Fkl', number_iter=2500, lr=0.003)
    
    mus_f_chi = train(detach=False, func='f', method='Chi', number_iter=1300, lr=0.003)
    mus_h_chi = train(detach=True, func='h', method='Chi', number_iter=1300, lr=0.003)
    
    mus_f_hel = train(detach=False, func='f', method='Hellinger', number_iter=3000, lr=0.003)
    mus_h_hel = train(detach=True, func='h', method='Hellinger', number_iter=3000, lr=0.003)
    
    save_plot(mus_h_rkl, mus_f_rkl, 'rkl')
    save_plot(mus_h_fkl, mus_f_fkl, 'fkl')
    save_plot(mus_h_chi, mus_f_chi, 'chi')
    save_plot(mus_h_hel, mus_f_hel, 'hel')