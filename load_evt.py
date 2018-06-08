import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import pdb
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class LoadDataset(Dataset):
	def __init__(self, years=range(2010,2019), horizon=4,lookback=24,transform=None):
		"""
		Args:
			years (list): List of years for which the data is to be loaded
		"""
		# years = range(2010,2019)
		# dfs = (pd.read_csv("http://pjm.com/pub/account/loadhryr/%s.txt"%str(yr)) for yr in years)
		dfs = (pd.read_csv("/Users/vsokolov/prj/grid/data/load/%s.txt"%str(yr),usecols=[0,1,2],names=["date","hour","load"],header=0) for yr in years)
		load_df   = pd.concat(dfs, ignore_index=True)
		load_df.date = pd.to_datetime(load_df.date, format='%m/%d/%y')
		load_df.hour = pd.to_numeric(load_df.hour/100, downcast="integer")
		load_df['dow'] = load_df.date.dt.dayofweek
		load_df['month'] = load_df.date.dt.month
		self.data = torch.tensor(load_df.sort_values(by=["date","hour"]).values[:,(1,3,4,2)].astype(int), device="cpu", dtype=torch.float32)
		# pdb.set_trace()
		self.std = self.data[:,3].std()
		self.mean = self.data[:,3].mean()
		self.min = self.data[:,3].min()
		self.max = self.data[:,3].max()
		# self.data[:,3] = (self.data[:,3] - self.min)/(self.max - self.min)
		self.data[:,3] = (self.data[:,3] - self.mean)/(self.std)
		self.horizon = horizon
		self.lookback = lookback
		self.transform = transform
	def __len__(self):
		return len(self.data)-self.lookback-self.horizon
	def __getitem__(self,idx):
		# pdb.set_trace()
		current = self.data[idx+self.lookback+self.horizon-1,3]
		# pdb.set_trace()
		# print(self.data[idx:(idx+self.lookback),3].size())
		# print(self.data[idx+self.lookback+self.horizon-1,0:3].size())
		# past    = torch.cat([self.data[idx:(idx+self.lookback),3],self.data[idx+self.lookback+self.horizon-1,0:3]])
		past    = torch.tensor(self.data[idx:(idx+self.lookback),3])
		# print(past.size())
		sample = {'past': past, 'current': current}
		if self.transform:
			sample = self.transform(sample)
		return sample	


transformed_dataset = LoadDataset()
dataloader = DataLoader(transformed_dataset, batch_size=32, num_workers=0)


# for i_batch, sample_batched in enumerate(dataloader):
# 	print(i_batch, sample_batched)
# 	if i_batch==3:
# 		break

class EVT(nn.Module):
	def __init__(self, nin, hidden_sizes, nout):
		super().__init__()
		self.nin = nin
		self.nout = nout
		self.hidden_sizes = hidden_sizes
		# pdb.set_trace()
		self.net = []
		hs = [nin] + hidden_sizes + [nout]
		for h0,h1 in zip(hs, hs[1:]):
			self.net.extend([nn.Linear(h0, h1),nn.ReLU()])
		self.net.pop() # pop the last nonlinear function for the output layer, just make it linear
		self.net = nn.Sequential(*self.net)
		# self.xi = torch.tensor([0.3], device=device, dtype=dtype, requires_grad=True)
	def forward(self, x):
		# pdb.set_trace()
		y = self.net(x)
		# return self.xi,y[:,0].exp()
		# return self.xi,y[:,0].clamp(min=1e-6)
		return y[:,0].clamp(1e-12)


dtype = torch.float
device = torch.device("cpu")
nin = 24
nout = 1
hidden = [30]
model = EVT(nin,hidden,nout)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
# optimizer = torch.optim.Adam(list(model.parameters()) + [model.xi],lr=learning_rate)
loss_func = torch.nn.MSELoss() 
# pdb.set_trace()
# u = (35000 - transformed_dataset.min)/(transformed_dataset.max - transformed_dataset.min)
u = (35500 - transformed_dataset.mean)/(transformed_dataset.std )
# u = 35000
print("u =",u)
nepoch = 3
for epoch in range(nepoch):
	for i, sample_batched in enumerate(dataloader):
		# sample_batched = next(enumerate(dataloader))[1]
		x,y = sample_batched["past"], sample_batched["current"]
		g = model(x)
		# tmp = (1+xi*(y-u)/s).clamp(min=1e-5)
		# loss = (s.log() + (xi.pow(-1)+1)*(tmp).log()).mean()
		
		indicator = (y-u) > 0
		indicator = indicator.type(dtype)
		ll = g.log() - g*(y-u)
		# pdb.set_trace()
		loss = -(ll*indicator).mean()
		# loss = loss_func(g, y) 
		if np.isnan(loss.item()):
			pdb.set_trace()
		# if (i%300==0):
		# 	print(i, loss.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print(epoch, loss.item())


# print(xi.item(),s.mean())
# pdb.set_trace()
# look at first few obsevations
n = 300
y = np.zeros(n)
yhat = np.zeros(n)
for i in range(n):
	yhat[i] = model(torch.tensor(transformed_dataset[i]["past"]).unsqueeze(0)).item()
	y[i] = transformed_dataset[i]["current"].item()
t = np.array(range(n))
plt.plot(t, transformed_dataset.std.item()*y+transformed_dataset.mean.item(), 'k--', t, transformed_dataset.std.item()*np.exp(u.item()*(1/yhat))+transformed_dataset.mean.item(), 'r-')
# plt.plot(t, transformed_dataset.std.item()*y+transformed_dataset.mean.item(), 'k--', t, transformed_dataset.std.item()*yhat+transformed_dataset.mean.item(), 'r-')
plt.show()


# model(torch.tensor(transformed_dataset[10]["past"]).unsqueeze(0))
# model(torch.tensor(transformed_dataset[40]["past"]).unsqueeze(0))

