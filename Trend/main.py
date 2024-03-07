from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import logging

import numpy as np
import xarray as xr
import pandas as pd
import time
import dask
import cmocean
import matplotlib.pyplot as plt 
import esmtools

from config import CHLConfig

cs = ConfigStore.instance()
cs.store(name="chl_config", node=CHLConfig)
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: CHLConfig) -> None:
    start_time = time.time()
    log.info(OmegaConf.to_yaml(cfg))

    #LOAD INPUT
    #######################
    if cfg.data.plot_only== False:
        if cfg.data.input == 'cmip':
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                data_input = '/home/datawork-lops-oh/biogeo/AI/CNN_CHLORO/CMIP6/IPSL-CM6A-LR/preprocess/'

                #Historical
                dsAH = xr.open_mfdataset(data_input + 'Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc')
                dsOH = xr.open_mfdataset(data_input + 'Omon_IPSL-CM6A-LR_historical_r1i1p1f1_gn_185001-201412.nc')
                #SSP
                dsAS = xr.open_mfdataset(data_input + 'Amon_IPSL-CM6A-LR_ssp585_r1i1p1f1_gr_201501-210012.nc')
                dsOS = xr.open_mfdataset(data_input + 'Omon_IPSL-CM6A-LR_ssp585_r1i1p1f1_gn_201501-210012.nc')
                ds_input = xr.merge([dsAS,dsOS,dsAH,dsOH])
                ds = ds_input[['chlos']]
                ds = ds.rename({'chlos':'chl'})

        if cfg.data.input == 'multiobs':
            data_input = '/home/datawork-lops-oh/biogeo/AI/CNN_CHLORO/MULTIOBS_GLO_BIO_BGC_3D_REP_015_010/Surface/'
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = xr.open_mfdataset(data_input + 'CMEMS_chl_*.nc')
                ds = ds.resample(time="1M").mean()
    
        if cfg.data.input == 'cci':
            data_input = '/home2/datawork/epauthen/Ocean-Colour-CCI/OC_CCI_Coarse/'
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = xr.open_mfdataset(data_input + "OC_CCI_chloro_a_*.nc")
                ds = ds.rename({'chlor_a_coarse': 'chl'})
                
        if cfg.data.input == 'globcolour_cmems':
            data_input = '/home/datawork-lops-oh/biogeo/AI/CNN_CHLORO/Globcolour_cmems_coarse/'
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = xr.open_mfdataset(data_input + "Globcolour_CMEMS_chl_*.nc")

        if cfg.data.input == 'yu':
            data_input = '/home/datawork-lops-oh/biogeo/AI/CNN_CHLORO/Chloro_Yu_2023_coarse/'
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = xr.open_mfdataset(data_input + "Yu_chloro_*.nc")

        if cfg.data.input == 'gsm':
            data_input = '/home2/datawork/epauthen/Globcolour_coarse/'
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                MODVIR    = xr.open_mfdataset(data_input + "MODVIR_*.nc")
                MERMOD    = xr.open_mfdataset(data_input + "MERMOD_*.nc")
                MERMODVIR = xr.open_mfdataset(data_input + "MERMODVIR_*.nc")
                MERMODSWF = xr.open_mfdataset(data_input + "MERMODSWF_*.nc")
                SWF    = xr.open_mfdataset(data_input + "SWF_*.nc")
                SWF    = SWF.sel(time = slice('1997-09','2002-06'))
                ds_gsm = MODVIR.merge(MERMOD)
                ds_gsm = ds_gsm.merge(MERMODSWF)
                ds_gsm = ds_gsm.merge(SWF)
                ds = ds_gsm.merge(MERMODVIR)
                ds = ds.rename({'CHL1_coarse': 'chl'})
        
        if cfg.data.input == 'avw':
            data_input = '/home2/datawork/epauthen/Globcolour_AVW_coarse/'
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                MODVIR_AV = xr.open_mfdataset(data_input + 'MODVIR_*.nc') 
                MERMOD_AV = xr.open_mfdataset(data_input + 'MERMOD_*.nc') 
                MERMODVIR_AV = xr.open_mfdataset(data_input + 'MERMODVIR_*.nc') 
                MERMODSWF_AV = xr.open_mfdataset(data_input + 'MERMODSWF_*.nc') 
                SWF    = xr.open_mfdataset(data_input + "SWF_*.nc")
                SWF    = SWF.sel(time = slice('1997-09','2002-06'))
                ds_avw = MODVIR_AV.merge(MERMOD_AV)
                ds_avw = ds_avw.merge(MERMODVIR_AV)
                ds_avw = ds_avw.merge(SWF)
                ds_avw = ds_avw.merge(MERMODSWF_AV)
                ds = ds_avw.rename({'CHL1_coarse': 'chl'})
        
        if cfg.data.input == 'GSMpred':
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = xr.open_dataset('/home/datawork-lops-oh/biogeo/AI/CNN_CHLORO/OUTPUT/Model_GRL/GSM_ensemble_mean.nc')
            ds = ds.rename({'chloro_pred': 'chl'})
            
        if cfg.data.input == 'CCIpred':
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = xr.open_dataset('/home/datawork-lops-oh/biogeo/AI/CNN_CHLORO/OUTPUT/Model_GRL/CCI_ensemble_mean.nc')
            ds = ds.rename({'chloro_pred': 'chl'})
            
        if cfg.data.input == 'GCMEMSpred':
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = xr.open_dataset('/home/datawork-lops-oh/biogeo/AI/CNN_CHLORO/OUTPUT/Model_GRL/GCMEMS_ensemble_mean.nc')
            ds = ds.rename({'chloro_pred': 'chl'})

        if cfg.data.input == 'VIRpred':
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = xr.open_dataset('/home/datawork-lops-oh/biogeo/AI/CNN_CHLORO/OUTPUT/Model_GRL/VIR_ensemble_mean.nc')
            ds = ds.rename({'chloro_pred': 'chl'})

        #Preprocessing
        ds_bathy = xr.open_dataset('/home2/datawork/epauthen/ETOPO1_Ice_g_gmt4.grd', engine='netcdf4')
        res = ds_bathy.z.interp(x=ds.longitude, y=ds.latitude,method = 'linear')
        ds = ds.assign(variables={"bathymetry": (('latitude','longitude'), res.data)})
        ds = ds.where(ds.bathymetry < -200)
        ds = ds.sel(time = slice(cfg.data.d1,cfg.data.d2))
        ds = ds.sel(latitude = slice(-50,50))
        ds = ds.drop(['bathymetry'])
        ds = ds.assign(variables={"chl_log": (('time','latitude','longitude'), np.log(ds.chl.data))})
        ds = ds.drop(['chl']).load()
        
        #count
        #ds.count(dim=['latitude','longitude']).chl_log.to_netcdf(cfg.path.path_save + "Count_" + str(cfg.data.input) +".nc")
        #trend on deseasonalised or raw
        ds_season = ds.groupby('time.month').mean(dim='time').chl_log
        ds_monthly = ds.groupby('time.month')
        ds = ds.assign(variables={"chl_deseason": (('time','latitude','longitude'), (ds_monthly - ds_monthly.mean(dim='time')).chl_log.data)})
        ds_lr = esmtools.stats.linregress(ds, dim='time', nan_policy='omit')        
        ds_lr.to_netcdf(str(cfg.path.path_save) + 
                        "Trend_log_chl_"+ str(cfg.data.d1)  + "_" + str(cfg.data.d2) + "_" +  str(cfg.data.input) +".nc")

    #Plot
    if cfg.data.plot:
        ds_lr = xr.open_dataset(str(cfg.path.path_save) + 
                                "Trend_log_chl_"+ str(cfg.data.d1)  + "_" + str(cfg.data.d2) + "_" +  str(cfg.data.input) +".nc")
        vm = 3
        w = 1200
#        if cfg.data.input in ('yu'):
#            w = 36525
#        if cfg.data.input in ('gsm', 'multiobs','globcolour_cmems','cci', 'avw'):
#            w = 1200

        fig = plt.figure(figsize=(7,4), dpi=180, facecolor='w', edgecolor='k')
        ds_not = ds_lr.where(ds_lr.sel(parameter='pvalue')>0.05)
        ds_sig = ds_lr.where(ds_lr.sel(parameter='pvalue')<=0.05)
        (ds_sig.chl_log.sel(parameter='slope')*w).plot(vmin = -vm,vmax = vm,cmap = cmocean.cm.balance)
    #    plt.contourf(ds_not.longitude,ds_not.latitude,ds_not.chl_log.sel(parameter='slope')*w,hatches=['....'], alpha=0, colors=None,levels=7)
        plt.title(str(cfg.data.input) + "(" + str(cfg.data.d1) + " to " + str(cfg.data.d2) + ")")
        plt.savefig(cfg.path.log + 'Fig_trend_'+ str(cfg.data.d1)  + "_" + str(cfg.data.d2) + "_" +  str(cfg.data.input) +'.png', bbox_inches='tight')

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    hours, remainder = divmod(elapsed_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    log.info(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
        
if __name__ == "__main__":
    main()

