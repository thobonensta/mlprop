#----------------------------------------------------------------------------------------------------------------------#
#                              FILE CONTAINING THE TRAINING MODEL FOR THE U-NET
#                  The training is performed using Adam Optimizer for a given metric
#                   The dataset used is composed of 2 to 5 obstacles (4000 samples)
#----------------------------------------------------------------------------------------------------------------------#
# Import (the AI model is build with torch)
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
from tqdm import tqdm
import argparse
from time import time
from aiModel import build_unet
import torch

def MSE_L1(output,target,alpha=0.6):
    ''' function that defines the loss function
    This latter is a combination of a L2 and L1 norm for regularization
    '''
    mse = criterion_mse(output, target)
    l1 = criterion_content(output, target)
    return alpha*mse+(1-alpha)*l1


def train(u,X,Y,batch_size,optimizer):
    ''' Training procedure over n epochs '''
    u.train()
    losse = 0
    total = 0
    # X is a torch Variable
    permutation = torch.randperm(X.shape[0])

    for i in tqdm(range(0,X.shape[0], batch_size)):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X[indices].to(device), Y[indices].to(device)

        outputs = u(batch_x)

        loss =  MSE_L1(outputs, batch_y)
        losse += loss.item()
        #print(loss)
        loss.backward()
        optimizer.step()

        total += 1

    train_losses.append(losse/total)

    np.savetxt(save_res+'/train_losses.txt',train_losses)


def test(u,X_test,Y_test,batch_size):
    ''' Function that computes the loss on the test data'''
    u.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(0,X_test.shape[0], batch_size)):
            batch_x, batch_y = X_test[i:i+batch_size].to(device), Y_test[i:i+batch_size].to(device)
            output = u(batch_x)
            test_loss +=  MSE_L1(output, batch_y).item()
            total += 1

    test_losses.append(test_loss/total)



#######################################################################################################
#
#                     Training of the model through execution.py
#               Here all the args of execution are used to train the model
#                  function train -> optimization of the weights
#                  function test -> test the network on test data
#
#######################################################################################################

var = int(round(time(),0))

# All the arguments given in execution.py
parser = argparse.ArgumentParser()
n_epochs = 40
batch_size = 8
lr = 0.001
b1 = 0.9
b2 = 0.999
eps = 1e-8
factor = 0.7
thresh = 1e-2
patience = 2
kernelD = 5
kernelU = 2
dilation_rate =12
diffI = True
dropoutP = 0.5
#
folder_path = "./"
directory = 'resTraining'
path = os.path.join(folder_path, directory)

try:
    os.makedirs(path,exist_ok=True)
    print("Directory '%s' created successfully" % directory)
except OSError as error:
    print("Directory '%s' can not be created")
save_res = path

# Create the logger that will store the important information
print('Initialize the UNet model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


u = build_unet(kernelSizeConv=kernelD,kernelSizeUp=kernelU,dilation=dilation_rate,diffI=diffI,p=dropoutP)
u = u.float()
u = u.to(device)
criterion_content = torch.nn.L1Loss()
criterion_mse = torch.nn.MSELoss()


# Training and testing dataset creation and shuffling

# Create the dataset
file = np.load('../data/dataField/data_2_obstacles_1000_v4.npz')
data1 = file['data']
data1 = np.array(data1)
file = np.load('../data/dataField/data_3_obstacles_1000_v4.npz')
data2 = file['data']
data2 = np.array(data2)
file = np.load('../data/dataField/data_4_obstacles_1000_v4.npz')
data3 = file['data']
data3 = np.array(data3)
file = np.load('../data/dataField/data_5_obstacles_1000_v4.npz')
data4 = file['data']
data4 = np.array(data4)

np.random.shuffle(data1)
np.random.shuffle(data2)
np.random.shuffle(data3)
np.random.shuffle(data4)

lim = 800 # 80/20 between train and validation

dataTrain1 = data1[:lim,:,:]
dataTrain2 = data2[:lim,:,:]
dataTrain3 = data3[:lim,:,:]
dataTrain4 = data4[:lim,:,:]
dataTrain = np.concatenate((dataTrain1,dataTrain2,dataTrain3,dataTrain4))
dataTest1 = data1[lim:,:,:]
dataTest2 = data2[lim:,:,:]
dataTest3 = data3[lim:,:,:]
dataTest4 = data4[lim:,:,:]
dataTest = np.concatenate((dataTest1,dataTest2,dataTest3,dataTest4))



#%%
# Optimizer
optimizer = optim.Adam(u.parameters(),lr=lr,betas=(b1, b2),eps=eps,weight_decay=0,amsgrad=False)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=factor,threshold=thresh,patience=patience, verbose=True)

#%%

# Creating the test dataset
dim = dataTrain.shape[2]
X = torch.from_numpy(dataTrain[:,0,:].reshape(-1,1,dim)).float()
Y = torch.from_numpy(dataTrain[:,1,:].reshape(-1,1,dim)).float()
X_test = torch.from_numpy(dataTest[:,0,:].reshape(-1,1,dim)).float()
Y_test = torch.from_numpy( dataTest[:,1,:].reshape(-1,1,dim)).float()

#%%
train_losses = []
train_accuracy = []
train_counter = []
test_losses = []
test_accuracy = []



# Learning process over n_epochs
for epoch in range(n_epochs):
    train(u,X,Y,batch_size,optimizer)
    test(u,X_test,Y_test,batch_size)


    scheduler.step(test_losses[-1])
    np.savetxt(save_res+'/train_losses.txt',train_losses)
    np.savetxt(save_res+'/test_losses.txt',test_losses)
    # saving model
    if len(test_losses) > 1 and test_losses[-1] < min(test_losses[:-1]):
        print('Save model at epoch '+str(epoch))
        model_path = os.path.join(path,"Model" + str(epoch) + ".pt")
        torch.save(u.state_dict(), model_path)



# Learning curves
plt.figure()
plt.title(f'Results from the training with lr={lr}')
epochs = [n+1 for n in range(n_epochs)]
plt.plot(epochs,train_losses,label='train loss')
plt.plot(epochs,test_losses, label='test loss')
plt.legend(loc='best')
plt.ylabel('Loss : MSE')
plt.xlabel('Epochs')
plt.savefig(save_res+'/results.png',bbox_inches='tight')









