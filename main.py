import os
import numpy as np
import pandas as pd

from ADEnv import ADEnv

# data path settings
data_path="/data/zhicao/UnknownAD"
data_folders=["NB15_unknown1"]
data_subsets={"NB15_unknown1":["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance"]}
# scenario settings
num_knowns=60
contamination_rate=0.02
# experiment settings
runs=1

# different datasets
for data_f in data_folders:
    # different unknown datasets for each dataset
    subsets=data_subsets[data_f]
    for subset in subsets:
        # location of unknwon datasets
        unknown_dataname="{}_{}_{}.csv".format(subset,contamination_rate,num_knowns)
        undata_path=os.path.join(data_path,data_f,unknown_dataname)
        # get unknown dataset
        table=pd.read_csv(undata_path)
        undataset=table.values

        # run experiment
        for _ in range(runs):
            env=ADEnv(undataset)
