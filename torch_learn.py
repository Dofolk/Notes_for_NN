#%%

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

#%%

def data(n):
    x = torch.unsqueeze(torch.linspace(-1,1,n), dim = 1)
    y = 2*x.pow(2) + 0.3 * torch.rand(x.size())
    return x,y

#%%
'''
x = torch.tensor([3.0])
y = torch.tensor([18.0])

a = torch.tensor([1.0], requires_grad=True)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD([a],lr=0.01)

for _ in range(100):
    optimizer.zero_grad()
    loss = loss_fn(y, a*x)
    loss.backward()
    optimizer.step()

print('a: ', a)
'''
class LearnML(nn.Module):
    def __init__(self, neurons):
        super(LearnML, self).__init__()
        self.net = self.network(neurons)

    def forward(self,x):
        res = self.net(x)
        return res
    
    def network(self, neuron):
        depth = len(neuron)
        layers = []
        for idx in range(1, depth):
            layer = nn.Linear(neuron[idx - 1],neuron[idx])
            layers.append(layer)
            if idx < depth:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def xavier_init(self):
        for name, param in self.net.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)
        return


# %%
if __name__ == '__main__':
    vec_len = 100
    x, y = data(vec_len)
    mdl = LearnML([1,vec_len,vec_len,1])
    optimizer = torch.optim.Adagrad(mdl.parameters())
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        pred = mdl(x.reshape(100,1))
        loss = loss_fn(pred,y.reshape(100,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%10 == 0:
            plt.scatter(x, mdl(x.reshape(100,1)).detach().numpy())
    

# %%
pred = mdl(x.reshape(100,1))

# %%
