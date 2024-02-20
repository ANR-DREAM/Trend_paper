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

## List of Data Used in This Study

| In Text Name      | File Name or Preprocessing        | Citation                | Link               |
| ----------------- | -------------------------------- | ----------------------- | ----------------------- |
| Yu2023            | -                                | Yu et al (2022)          |  https://doi.org/10.5281/zenodo.7092220 |
| Multiobs-CMEMS    | MULTIOBS-GLO-BIO-BGC-3D-REP-015-010 | Sauzede et al 2021      |
| OC-CCI            | OCEANCOLOUR-GLO-BGC-L3-MY-009-107  | ESA 2023          |
| Globcolour-CMEMS  | OCEANCOLOUR-GLO-BGC-L3-MY-009-103  | ACRI 2023         |
| Globcolour-AVW    | -                                | ACRI 2020         |
| Globcolour-GSM    | -                                | ACRI 2020         |
| SeaWiFS           | GSM and NASA R2018.0             | ACRI 2020         |
| VIIRS-NPP         | GSM and NASA R2018.0             | ACRI 2020         |
| MODIS AQUA        | GSM and NASA R2018.0             | ACRI 2020         |
| OLCI-A            | GSM and ESA PB 2.16 to 2.55      | ACRI 2020         |
| MERIS             | GSM and ESA third preprocessing  | ACRI 2020         |
| SST               | METOFFICE-GLO-SST-L4-REP-OBS-SST | Worsfold et al (2022)    |
| SLA               | SEALEVEL-GLO-PHY-L4-MY-008-047   | CLS 2023         |
| U10, V10, SSR     | ERA5                             | Hersbach et al 2023     |
| U, V              | OSCAR-L4-OC-FINAL-V2.0           | Dohan et al 2021       |
| MDT               | CNES-CLS18                        | Mulet et al 2021       |

