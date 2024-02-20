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
\begin{table}[]
\caption{List of data used in this study, for the six Schl merged products, the five Schl mono sensors and for the eight input features of the CNN.}
\begin{tabular}{ccc}
In text name           & File name or preprocessing          & Citation                                   \\ \hline
Yu2023    & -                                   & \cite{Yu:2022aa}                      \\
Multiobs-CMEMS   & MULTIOBS-GLO-BIO-BGC-3D-REP-015-010 & \cite{Sauzede:2021ab}                  \\
OC-CCI           & OCEANCOLOUR-GLO-BGC-L3-MY-009-107   & \cite{ESA:2023aa}                          \\
Globcolour-CMEMS & OCEANCOLOUR-GLO-BGC-L3-MY-009-103   & \cite{ACRI:2023aa}                          \\
Globcolour-AVW   & -                                   & \cite{ACRI:2020aa}                              \\
Globcolour-GSM   & -                                   & \cite{ACRI:2020aa}                          \\ \hline
SeaWiFS          & GSM and NASA R2018.0                & \cite{ACRI:2020aa}                              \\
VIIRS-NPP        & GSM and NASA R2018.0                & \cite{ACRI:2020aa}                             \\
MODIS AQUA       & GSM and NASA R2018.0                & \cite{ACRI:2020aa}                            \\
OLCI-A           & GSM and ESA PB 2.16 to 2.55         & \cite{ACRI:2020aa}                              \\
MERIS            & GSM and ESA third preprocessing     & \cite{ACRI:2020aa}                             \\
\hline
SST                    & METOFFICE-GLO-SST-L4-REP-OBS-SST    & \cite{Worsfold:2022aa}                       \\
SLA                    & SEALEVEL-GLO-PHY-L4-MY-008-047      & \cite{CLS:2023aa}                          \\
U10, V10, SSR          & ERA5                                & \cite{Hersbach:2023aa}                       \\
U, V                   & OSCAR-L4-OC-FINAL-V2.0              & \cite{Dohan:2021aa}                         \\
MDT                    & CNES-CLS18                          & \cite{Mulet:2021aa}             
\end{tabular}
\end{table}
