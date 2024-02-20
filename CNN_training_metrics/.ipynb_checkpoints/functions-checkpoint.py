import os
import dask
import cmocean
import statsmodels
import numpy as np
import xarray as xr
import pandas as pd
import colorcet as cc
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, DataLoader
import torch.optim as optim
import torch

def train_network(model,dataloaders, criterion ,n_epochs, lr, path_log):
    """
    Train a neural network
    
    Parameters
    ----------
    model : model class object
    dataloaders : tensor Pytorch object containing the X_train, X_valid, Y_train and Y_valid
    criterion : specify the loss function (e.g. torch.nn.MSELoss())
    n_epochs : number of epoch to run
        
    Returns
    -------
    model.pt : the weight of the model stored
    Loss.png : plot of the train and validation loss curve
    train_losses_save.npy : loss values for the train dataset (as plotted on Loss.png)
    valid_losses_save.npy : loss values for the validation dataset (as plotted on Loss.png)
    """ 

    #Get optimizer
    optimizer = optim.Adam(model.parameters(), lr = lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float)
    train_losses_save, valid_losses_save = [], [] 
    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        
        model.train()
        for data, target in dataloaders['train']:
            data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.float)
            optimizer.zero_grad()

            output = model(data.float())
            output2 = output
            target2 = target

            # Apply the mask land to compute the loss only on the ocean
            mask = np.isnan(target.cpu())
            mask = mask.bool()
            mask = ~mask            
            mask = mask.to(device)
        
            output2 = torch.masked_select(output2, mask)
            target2 = torch.masked_select(target2, mask)           
            # calculate the batch loss
            loss = criterion(output2.float(), target2.float())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
                    
        model.eval()
        for data, target in dataloaders['val']:
            data,target=data.to(device,dtype=torch.float),target.to(device,dtype=torch.float)

            output = model(data.float())            
            output2 = output
            target2 = target
            
            # Apply the mask land to compute the loss only on the ocean
            mask = np.isnan(target.cpu())
            mask = mask.bool()
            mask = ~mask
            mask = mask.to(device)
        
            output2 = torch.masked_select(output2, mask)
            target2 = torch.masked_select(target2, mask)
            
            loss = criterion(output2.float(), target2.float())
            valid_loss += loss.item()
        
        # calculate average losses and store
        train_loss = train_loss/len(dataloaders['train'].sampler)
        valid_loss = valid_loss/len(dataloaders['val'].sampler)
        train_losses_save.append(train_loss)
        valid_losses_save.append(valid_loss)

        # print training/validation statistics 
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, n_epochs, train_loss, valid_loss))

    # Plot the loss curve
    fig = plt.subplots(1, 1, figsize=(5, 5),sharey = True, tight_layout=True)
    plt.plot(valid_losses_save, label='Validation loss',color='orange')
    plt.plot(train_losses_save, label='Training loss',color='blue', linestyle='dashed')
    plt.legend(frameon=False)
    plt.savefig(path_log +'Loss', dpi= 100,bbox_inches = "tight")
    
    # Save loss values
    np.save(path_log +'train_loss.npy',train_losses_save)
    np.save(path_log +'valid_loss.npy',valid_losses_save)

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model.'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), path_log + 'model.pt')   
        valid_loss_min = valid_loss
    del dataloaders

    
def train_val_dataset(X_scaled, Y_scaled, time, val_split, batch_size, log,random_state):
    """
    Split between train and validation function (20% of validation data by default)
    
    Parameters
    ----------
    X : input data scaled
    Y : output data scaled
    time : vector of time to use for the plot of train/validation distribution
    val_split : percentage of data to take in the validation split
    batch_size : size of the batch
    log : path of the working directory to save plots and files
    
    Returns
    -------
    dataloaders : data object containing a split "train" and "val"
    """ 
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Dataset_XY(X = X_scaled, Y = Y_scaled, transform=transform)
    if random_state=='None':
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    if isinstance(random_state, int):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split,random_state=random_state)
    if random_state=="stratify":
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, stratify=time.dt.month)

    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    dataloaders = {x:DataLoader(datasets[x],batch_size, shuffle=True, num_workers=0) for x in ['train','val']}
    next(iter(dataloaders['train']))
    x,y = next(iter(dataloaders['train']))

    #Plot the train and validation separation in time
    time_train = time[train_idx]
    time_valid = time[val_idx]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5),sharey = True, tight_layout=True)
    ax[0].hist([time_train,time_valid],label = ['train (' + str(len(time_train)) + ')','val (' + str(len(time_valid)) + ')'])
    ax[0].legend()
    ax[1].hist([time_train.dt.month,time_valid.dt.month],bins = 12)
    plt.savefig(log + 'Train_valid_time', dpi= 100,bbox_inches = "tight")
    plt.close()
    del X_scaled
    del Y_scaled
    return dataloaders


class Dataset_XY(Dataset):
    
    def __init__(self, X, Y , transform=None):
        """
        Args:
            root_folder (String): path to input and output files
            input_file (String): npy array of the input data to be used by the NN
            output_file (String): npy array of the output data to be use by the NN
            transform (callable, Optional): Optional transform to be applied on
            a sample
            The size of input are by default : time x lat x lon x variable
            The size of output are by default : time x lat x lon
        """
        self.input_arr = X
        self.output_arr = Y
        self.transform = transform
        
    def __len__(self):
        return self.input_arr.shape[0]
    
    def __getitem__(self, idx):
        X = self.input_arr[idx,...]
        Y = self.output_arr[idx,...]
        
        if self.transform:
            X =  self.transform(X)
            Y =  self.transform(Y)
        
        return X,Y
    
class Dataset_X(Dataset):
    def __init__(self, X , transform=None):
        """
        Args:
            root_folder (String): path to input and output files
            input_file (String): npy array of the input data to be used by the NN
            output_file (String): npy array of the output data to be use by the NN
            transform (callable, Optional): Optional transform to be applied on
            a sample
            The size of input are by default : time x lat x lon x variable
            The size of output are by default : time x lat x lon
        """
        self.input_arr = X
        self.transform = transform
        
    def __len__(self):
        return self.input_arr.shape[0]
    
    def __getitem__(self, idx):
        X = self.input_arr[idx,...]
        
        if self.transform:
            X =  self.transform(X)
        
        return X

def chloro_prediction(ds_input,model,log,d1_pred,d2_pred,Xm,Xstd,Cm,Cstd,var_input):
    """
    Compute predictions of chlorophyl from ds_input, using the model.
    
    Parameters
    ----------
    ds_input : input data
    log : path of the working directory to save plots and files
    d1,d2 : test period interval
    Xm, Xstd,Cm, Cstd : Scalers for input and for chlorophyl, computed on the train dataset.
    var_input : list of variable inputs
    
    Returns
    -------
    Chloro_pred_time.nc : netcdf Xarray files of chlorophyl maps predicted by months
    """ 
    torch.cuda.empty_cache() 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(log + 'Chloro_pred/'):
        os.makedirs(log + 'Chloro_pred/')
    #Load the weight of the trained model
    model.load_state_dict(torch.load(log + 'model.pt',map_location=torch.device('cpu')))    
    
    x_test = ds_input.sel(time = slice(d1_pred,d2_pred))
    x_test   = x_test.fillna(0)
    x_scaled = (x_test - Xm)/Xstd

    #Create X ndarray and fill with scaled variables
    n_channel = len(var_input)
    X_scaled = np.zeros([x_scaled.time.size,
                         x_scaled.latitude.size,
                         x_scaled.longitude.size,
                         n_channel], dtype='float32')
    i = 0
    for v in var_input:
        X_scaled[:,:,:,i] = x_scaled[v].data.astype('float32')
        i+=1

    #Predict chloro months by months
    for i in np.arange(0,len(x_test.time)):
        X_batch = np.expand_dims(X_scaled[i,:,:,:],axis = 0)
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = Dataset_X(X = X_batch, transform=transform)
        test_loader = DataLoader(dataset, batch_size = 1, sampler=None, num_workers=0)

        #Prediction and save test chlorophyl maps
        for inputs in test_loader:
            inputs = inputs.to(device,dtype=torch.float)
            outputs = model(inputs.float()).detach()
            chl = outputs.cpu().numpy().squeeze() 
            chl = chl*Cstd.compute().data + Cm.compute().data
            chl = np.exp(chl)

            #Assign prediction to dataset, remove land mask and save
            y_pred = xr.Dataset(
                    data_vars={'chloro_pred':    (('time', 'latitude', 'longitude' ), np.expand_dims(chl, 0))},
                    coords={ 'time':np.atleast_1d(x_test.time.isel(time = i)),
                            'longitude': x_test.longitude,
                            'latitude': x_test.latitude})
            y_pred = y_pred.where(ds_input.mask == 1)
            y_pred.to_netcdf(log +'Chloro_pred/Chloro_pred_' + str(y_pred.isel(time = 0).time.data)[0:10] + '.nc')
            y_pred.close()

def test_metrics(ds_out,log,d1_test, d2_test):
    """
    Compute test metrics.
    
    Parameters
    ----------
    ds_out : output data to compare with 
    log : path of the working directory to load prediction and save plots and files
    d1,d2 : test period interval
    
    Returns
    -------
    Metrics.csv : Table of metrics computed 
    """ 
    y_test = ds_out.sel(time = slice(d1_test,d2_test)).load()

    y_pred = xr.open_mfdataset(log +'Chloro_pred/Chloro_pred_*.nc')
    y_pred = y_pred.sel(time = slice(d1_test,d2_test)).load()
    mask = np.isnan(y_test.chloro) #Remove predicted Clouds and missing data
    mask = ~mask            
    y_pred = y_pred.where(mask == 1)
    
    #Stack and dropna
    obs = y_test.chloro.stack(n_prof=("longitude", "latitude","time"))
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        obs = obs.dropna(dim = "n_prof",how="all")

    pred = y_pred.chloro_pred.stack(n_prof=("longitude", "latitude","time"))
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        pred = pred.dropna(dim = "n_prof",how="all")
    
    obs  = np.log(obs)
    pred = np.log(pred)
    
    # Replace -inf with -10 (for CanESM5 CMIP model)
    obs = obs.where(~np.isinf(obs), other=-10)

    reg = stats.linregress(obs,pred)
    slope     = reg.slope.round(2)
    intercept = reg.intercept.round(2)
    rvalue    = reg.rvalue
    r2        = (rvalue**2).round(2)
    rmse      = np.sqrt(((obs - pred) ** 2).mean()).round(2).compute().data
    mae       = np.abs((obs - pred)).mean().round(2).compute().data

    #Display
    df = pd.DataFrame([r2,rmse,mae,slope], columns=['Test ('+d1_test+' to '+d2_test+')']
                , index=["R$^{2}$", "RMSE", "MAE", "Regression Slope"])
    df.to_csv(log +'Metrics.csv')


def plot_trend(ds_out, output, input_name, log,d1_pred,d2_pred,d1_train,d2_train,t):
    """
    Plot trends of chlorophyl observed or modeled and predicted by the NN.
    Plot one timestep map of chlorophyl observed or modeled and predicted by the NN.
    
    Parameters
    ----------
    ds_out : output data to compare with 
    y_pred : chlorophyl predicted 
    log : path of the working directory to load prediction and save plots and files
    d1_pred,d2_pred : period interval predicted, to plot
    d1_train,d2_train : period interval trained on
    t : date to be plotted on the mpa
    
    Returns
    -------
    Trend_mean.png, Trend_median.png : one with the median trend nad one with the mean.
    Chloro_map_t.png : Map of chloro observed, chloro predicted by the NN, difference of the two.
    """ 

    y_pred = xr.open_mfdataset(log +'Chloro_pred/Chloro_pred_*.nc')
    y_pred = y_pred.sel(time = slice(d1_pred,d2_pred)).load()
    y_test = ds_out.sel(time = slice(d1_pred,d2_pred)).load()
    y_test = y_test.where(ds_out.mask == 1)

    # Compute weighted mean by timestep for test data
    weights = np.cos(np.deg2rad(y_test.latitude))
    weights.name = "weights"
    y_testw = y_test.chloro.weighted(weights)
    chloro_qua = y_testw.quantile(dim = ("longitude", "latitude"),q=0.5)
    chloro_mean = y_testw.mean(dim = ("longitude", "latitude"))
    y_test = y_test.assign(variables={"chloro_qua": (('time'), chloro_qua.data)}) 
    y_test = y_test.assign(variables={"chloro_mean": (('time'), chloro_mean.data)}) 

    #Clouds/missing data
    mask = np.isnan(y_test.chloro) 
    # Ensure masking of period outside of output period.
    if output == "obs_globcolour":     
        mask1 = xr.DataArray(False, dims=("longitude","latitude","time"), 
                             coords={"longitude": mask.longitude, "latitude": mask.latitude, 
                                     "time": y_pred.time.sel(time= slice(d1_pred,y_test.time[0]-1))})
        mask = xr.concat([mask, mask1], dim="time")

    y_pred = y_pred.where(mask == False)
    
    #Mean and median for prediction
    weights = np.cos(np.deg2rad(y_pred.latitude))
    weights.name = "weights"
    y_predw = y_pred.chloro_pred.weighted(weights)
    chloro_mean = y_predw.mean(dim = ['longitude','latitude'])
    chloro_qua = y_predw.quantile(dim = ['longitude','latitude'],q = 0.5)

    y_pred_mean = xr.Dataset(coords={'time': y_pred.time})
    y_pred_qua = xr.Dataset(coords={'time': y_pred.time})
    y_pred_mean = y_pred_mean.assign(variables={"chloro_mean": (('time'), chloro_mean.data)})
    y_pred_qua = y_pred_qua.assign(variables={"chloro_qua": (('time'), chloro_qua.data)})
        
    y_pred_mean.to_netcdf(log + 'pred_mean.nc')
    y_pred_qua.to_netcdf(log + 'pred_median.nc')

    #Plots of Median and Mean
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')
    y_pred_mean.chloro_mean.plot(ax=ax[0],label = 'pred mean ',color = 'tab:red')
    y_test.chloro_mean.plot(ax = ax[0],label = input_name + ' mean',color = 'black')

    y_pred_qua.chloro_qua.plot(ax=ax[1],label = 'pred median',color = 'tab:red')
    y_test.chloro_qua.plot(ax = ax[1],label = input_name + ' median',color = 'black')
    for i in np.arange(2):
        ax[i].legend()
        ax[i].grid()
        ax[i].set_xlim([pd.to_datetime(d1_pred),pd.to_datetime(d2_pred)])
        ax[i].axvline(x = pd.to_datetime(d1_train),linestyle = '--',color = 'grey')
        ax[i].axvline(x = pd.to_datetime(d2_train),linestyle = '--',color = 'grey')
    plt.savefig(log +'Mean_Median', dpi= 100,bbox_inches = "tight")

    #Plot the maps
    y_test = y_test.sel(time=t)
    y_pred = y_pred.sel(time=t)
    cm = cc.cm["rainbow"]
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,3), dpi=80, facecolor='w', edgecolor='k')
    y_pred.chloro_pred.plot(ax = ax[1],cmap = cm,norm=colors.LogNorm(vmin=0.01, vmax=10))
    y_test.chloro.plot(ax = ax[0],cmap = cm,norm=colors.LogNorm(vmin=0.01, vmax=10))
    
    ax[0].set_title('Chloro '+input_name+', ' + str(y_pred.time.data).split('T')[0])
    ax[1].set_title('Chloro pred, ' + str(y_pred.time.data).split('T')[0])

    (y_pred.chloro_pred-y_test.chloro).plot(ax = ax[2],cmap = cmocean.cm.balance,vmin = -1,vmax = 1)
    ax[2].set_title('Chloro pred - '+input_name + ' ' + str(y_pred.time.data).split('T')[0])
    plt.savefig(log +'Chloro_map_' + t, dpi= 100,bbox_inches = "tight")


