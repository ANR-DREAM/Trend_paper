# Contrasted trends in chlorophyll-a satellite products
Codes necessary to produce the results of the paper "Contrasted trends in chlorophyll-a satellite products" Pauthenet et al (submitted)
The training of the CNN, the reconstruction of chlorophyll-a fields and computation of metrics, along with some diagnostic plots are implmeneted with Hydra.cc.
To install the python environment first install [Micromamba](https://mamba.readthedocs.io/en/latest/micromamba-installation.html) (```brew install micromamba``` on MacOS), then install your environment :
```
micromamba env create -f environment.yml 
conda activate EP-2023.01
```
Finally use the config.yaml file to select data and model parameters, then run ```python main.py``` or the bash script ```./run_script.sh``` to get several run (bootstrap).  

Python notebooks are in the folder Figure to produce each of the four figures of the paper. 

The data is available following these URLs and DOIs.
