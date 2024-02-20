# Contrasted trends in chlorophyll-a satellite products
Codes necessary to produce the results of the paper "Contrasted trends in chlorophyll-a satellite products" Pauthenet et al (submitted).

The training of the CNN, the reconstruction of chlorophyll-a fields and computation of metrics, along with some diagnostic plots are implmeneted with Hydra.cc.
To install the python environment first install [Micromamba](https://mamba.readthedocs.io/en/latest/micromamba-installation.html) (```brew install micromamba``` on MacOS), then install your environment :
```
micromamba env create -f environment.yml 
conda activate EP-2023.01
```
Finally use the config.yaml file to select data and model parameters, then run ```python main.py``` or the bash script ```./run_script.sh``` to get several run (bootstrap).  

Python notebooks are in the folder Figure to produce each of the four figures of the paper. 

## List of data used in this study

| In Text Name      | File Name or Preprocessing        | Citation                | Link               |
| ----------------- | -------------------------------- | ----------------------- | ----------------------- |
| Yu2023            | -                                | Yu et al 2022          |  https://doi.org/10.5281/zenodo.7092220 |
| Multiobs-CMEMS    | MULTIOBS-GLO-BIO-BGC-3D-REP-015-010 | Sauzede et al 2021      | https://doi.org/10.48670/moi-00046 |
| OC-CCI            | OCEANCOLOUR-GLO-BGC-L3-MY-009-107  | ESA 2023          | https://doi.org/10.48670/moi-00282 |
| Globcolour-CMEMS  | OCEANCOLOUR-GLO-BGC-L3-MY-009-103  | ACRI-ST 2023         | https://doi.org/10.48670/moi-00280 |
| Globcolour-AVW    | -                                | ACRI-ST 2020         | https://hermes.acri.fr |
| Globcolour-GSM    | -                                | ACRI-ST 2020         | https://hermes.acri.fr |
| SeaWiFS           | GSM and NASA R2018.0             | ACRI-ST 2020         | https://hermes.acri.fr |
| VIIRS-NPP         | GSM and NASA R2018.0             | ACRI-ST 2020         | https://hermes.acri.fr |
| MODIS AQUA        | GSM and NASA R2018.0             | ACRI-ST 2020         | https://hermes.acri.fr |
| OLCI-A            | GSM and ESA PB 2.16 to 2.55      | ACRI-ST 2020         | https://hermes.acri.fr |
| MERIS             | GSM and ESA third preprocessing  | ACRI-ST 2020         | https://hermes.acri.fr |
| SST               | METOFFICE-GLO-SST-L4-REP-OBS-SST | Worsfold et al 2022    | https://doi.org/10.48670/moi-00168 |
| SLA               | SEALEVEL-GLO-PHY-L4-MY-008-047   | CLS 2023         |    https://doi.org/10.48670/moi-00148 |
| U10, V10, SSR     | ERA5                             | Hersbach et al 2023     | [https://doi.org/10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47) |
| U, V              | OSCAR-L4-OC-FINAL-V2.0           | Dohan et al 2021       | https://doi.org/10.5067/OSCAR-25F20 |
| MDT               | CNES-CLS18                        | Mulet et al 2021       | [AVISO](https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/mdt/mdt-global-cnes-cls18.html) |

## Data reference
ACRI-ST. (2020). Globcolour chlorophyll concentration for case 1 waters avw, gsm [dataset]. Author. Retrieved from https://hermes.acri.fr

ACRI-ST, Colella, S., Böhm, E., Cesarini, C., Garnesson, P., J.Netting, & Calton, B. (2023). Global ocean colour (copernicus-globcolour), bio-geo-chemical, l3 (daily) from satellite observations (1997-ongoing) [dataset]. E.U. Copernicus Marine Service Information (CMEMS). Marine Data Store (MDS). Retrieved from https://doi.org/10.48670/moi-00280 doi: 10.48670/moi-00280

CLS, & Pujol, M.-I. (2023). Global ocean gridded l 4 sea surface heights and derived variables reprocessed 1993 ongoing [dataset]. E.U. Copernicus Marine Service Information (CMEMS). Marine Data Store (MDS). Retrieved from https://doi.org/10.48670/moi-00148 doi: 10.48670/moi-00148

Dohan, K. (2021). Ocean surface current analyses real-time (oscar) surface currents - final 0.25 degree (version 2.0). NASA Physical Oceanography Distributed Active Archive Center. Retrieved from https://podaac.jpl.nasa.gov/dataset/OSCAR L4 OC FINAL V2.0 doi: 10.5067/OSCAR-25F20

ESA, Colella, S., Böhm, E., Cesarini, C., Garnesson, P., J.Netting, & Calton, B. (2023). Global ocean colour plankton and reflectances my l3 daily observations [dataset]. E.U. Copernicus Marine Service Information (CMEMS). Marine Data Store (MDS). Retrieved from https://doi.org/10.48670/moi-00282 doi: 10.48670/moi-00282

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horanyi, A., Munoz Sabater, J.,. . . Thépaut, J.-N. (2023). Era5 hourly data on single levels from 1940 to present. [dataset]. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). Retrieved from https://doi.org/10.24381/cds.adbb2d47 doi: 10.24381/cds.adbb2d47

Mulet, S., Rio, M.-H., Etienne, H., Artana, C., Cancet, M., Dibarboure, G., . . . Provost, C. (2021). The new cnes-cls18 global mean dynamic topography. Ocean Science, 17(3), 789–808.

Sauzede, R., Renosh, P., & Claustre, H. (2021). Global ocean 3d chlorophyll-a con- centration, particulate backscattering coefficient and particulate organic carbon [dataset]. E.U. Copernicus Marine Service Information (CMEMS). Marine Data Store (MDS). Retrieved from https://doi.org/10.48670/moi-00046 doi: 10.48670/moi-00046

Worsfold, M., Good, S., Martin, M., McLaren, A., Roberts-Jones, J., Fiedler, E., & Met Office, U. (2022). Global ocean ostia sea surface temperature and sea ice reprocessed [dataset]. E.U. Copernicus Marine Service Information (CMEMS). Marine Data Store (MDS). Retrieved from https://doi.org/10.48670/moi-00168 doi: 10.48670/moi-00168
