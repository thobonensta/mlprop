import time

import numpy as np
import torch
import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt
from aiModel.aiModel import build_unet

# losses definition
criterion_content = torch.nn.L1Loss()
criterion_mse = torch.nn.MSELoss()

def MSE(output, target):
    mse = criterion_mse(output, target)
    return mse

def MAE(output, target):
    l1 = criterion_content(output, target)
    return l1

#######################################################################################################
#
#                       LOAD THE DATA TO TEST THE MODEL
#
#######################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du périphérique : {device}")


# Data with 6 obstacles to validate
file = np.load('./data/dataField/data_6_obstacles_100.npz')
data_6_50 = file['data']
dim = data_6_50.shape[2]
X6_ = torch.from_numpy(data_6_50[::,0,:].reshape(-1,1,dim)).float().to(device)
Y6_ = torch.from_numpy(data_6_50[::,1,:].reshape(-1,1,dim)).float().to(device)


# Example of terrain data with 6 obstacles
idx = 89
f, ax = plt.subplots(figsize=(5, 4))
x = [0.050 * i for i in range(dim)]
plt.plot(x, X6_[idx].cpu().reshape((-1,)), label='Relief')
plt.xlabel('Distance (km)')
plt.ylabel('Altitude (m)')
plt.legend()
plt.show()


#######################################################################################################
#
#                            CREATE (LOAD) THE U-NET MODEL
#
#######################################################################################################
# Weighted combination for the loss
u_comb_1 = build_unet(kernelSizeConv=5,dilation=12,diffI=True,p=0.5)
u_comb_1 = u_comb_1.float().to(device)
u_comb_1.load_state_dict(torch.load('./aiModel/ModelSaved/results_1743841158/model36.pt',map_location=torch.device(device)))
u_comb_1.eval()


#######################################################################################################
#
#                          TEST THE MODEL OVER THE 6 OBSTACLES DATA
#
#######################################################################################################
from tqdm import tqdm
batch = 8
with torch.no_grad():
    loss1= 0
    compt = 0
    for i in tqdm(range(0,X6_.shape[0], batch)):
        batch_x, batch_y = X6_[i:i+batch], Y6_[i:i+batch]
        outputs1 = u_comb_1(batch_x)


        loss1 += MSE(outputs1,batch_y).item()

        compt+=1

L2_comb1 = loss1/compt


print(L2_comb1)

#
# # Plot for one sample
t_0 = time.perf_counter()
ouput_50_comb = u_comb_1(X6_[idx].reshape(1, 1, dim))
#torch.cuda.synchronize()
print('Inference time : ',time.perf_counter()-t_0)
ouput_50 = u_comb_1(X6_[idx].reshape(1, 1, dim))

# mse_50 = round(MSE(ouput_50, Y6_[idx].reshape(1, 1, dim)).item(), 2)


# # print(f'Inference time : {dt}')
x_50 = np.array([50 * i for i in range(dim)]) / 1000
f, ax = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 2]})
# first subplot : the terrain data
ax[0].plot(x_50, X6_[idx].cpu().numpy().reshape((-1,)), color='black', label='Terrain relief')
ax[0].fill_between(x_50, X6_[idx].cpu().numpy().reshape((-1,)), where=X6_[idx].cpu().numpy().reshape((-1,)) / 5 >= 0, facecolor='black')
ax[0].grid(axis='both')
ax[0].set_xticks([])
ax[0].set_ylabel('Relief (m)')
ax[0].set_ylim([0, 70])
ax[0].set_xlim([0, 75])
# second subplot the target field (db) and the prediction with the model
ax[1].plot(x_50, Y6_[idx].cpu().numpy().reshape((-1,)), label='Target', alpha=0.8)
ax[1].plot(x_50, ouput_50.cpu().detach().numpy().reshape((-1,)), '--', label=f'MSE loss')
ax[1].plot(x_50, ouput_50_comb.cpu().detach().numpy().reshape((-1,)), '--', label=f'Combined loss')

ax[1].grid(axis='both')
ax[1].set_xlabel('Distance (km)')
ax[1].set_ylabel('Field u (dB)')
ax[1].set_xlim([0, 75])
plt.legend()
plt.show()
