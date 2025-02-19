# Data and code for *"Higher-order shortest paths in hypergraphs"*

This repository contains data and code to analysed in the following paper:
*Berné L. Nortier, Simon Dobson and Federico Battiston ["Higher-order shortest paths in hypergraphs"](https://arxiv.org/abs/2502.03020), arXiv Preprint (2025)*


# Data
The data that support the findings of this study are available at:
* **SocioPatterns** datasets (InVS15, LH10, SFHH, LyonSchool, Thiers13) by the [SocioPatterns project](http://www.sociopatterns.org/). Data source [here](http://www.sociopatterns.org/datasets/);
* **Utah’s schools** datasets (Mid1, Elem1) by the Contacts among Utah's School-age Population (CUSP), presented in [Toth et al. J. R. Soc. Interface **12**: 20150279 (2015)](https://royalsocietypublishing.org/doi/10.1098/rsif.2015.0279). Data source [here](https://royalsocietypublishing.org/doi/suppl/10.1098/rsif.2015.0279);
* **Friends & Family** dataset (FnF_2010-10 - FnF_2011-06) is not licensed for redistribution so we have not included them here, but they can be requested from [here](http://realitycommons.media.mit.edu/friendsdataset.html); 
* **Copenhagen** dataset (Copenhagen) from the Copenhagen Networks Study presented in [Sapiezynski et al. (2019).](https://doi.org/10.6084/m9.figshare.7267433.v1) Data source [here](https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433/1);
* **Malawi** dataset (Malawi) from [Ozella, L., Paolotti, D., Lichand, G. et al. EPJ Data Sci. 10, 46 (2021)](https://doi.org/10.1140/epjds/s13688-021-00302-w). Data source [here](http://www.sociopatterns.org/wp-content/uploads/2021/09/tnet_malawi_pilot.csv.gz);
* **Kenyan** dataset (Kenya) from [Kiti, M.C. et al. EPJ Data Sci. 5, 21 (2016)](https://doi.org/10.1140/epjds/s13688-016-0084-2). The data can be found [here](http://www.sociopatterns.org/datasets/kenyan-households-contact-network/) 
* **Baboons** datase (baboons) from [Gelardi V. et al. Proc. R. Soc. A.47620190737 (2020)](http://doi.org/10.1098/rspa.2019.0737). The data can be found [here](http://www.sociopatterns.org/datasets/baboons-interactions/)
* **Political interactions** datasets (congress-bills), presented in [P. S. Chodrow et al., Science Advances **7**, eabh1303 (2021)](https://www.science.org/doi/10.1126/sciadv.abh1303).


# Code

For loading and creating temporal networks, **2 python scripts and a helper file** are provided:
* The file `preprocessing.ipynb` creates all temporal and static networks from the raw data and stores them;
* All helper functions are available in `utils_preprocessing.py`;
* Finally, `FnF_aggregation_window.ipynb` plots the size of the largest connected component as a function of the aggregation window in minutes for the Friends & Family dataset. 


The `Data` folder will contain all raw and processed data. 

The scripts assume the following folder hierarchy:

```
Data
├── dataset1/
│   ├── raw/
│   ├── processed/
...
└── datasetK/
    ├── raw/
    └── processed/
```
where raw data is stored in `raw` and generated temporal and static networks are stored in `processed`.



