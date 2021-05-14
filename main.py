import os
import numpy as np
import pandas as pd

from ADEnv import ADEnv

### Basic Settings
# data path settings
data_path="/data/zhicao/UnknownAD"
data_folders=["NB15_unknown1"]
data_subsets={"NB15_unknown1":["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance"]}
# scenario settings
num_knowns=60
contamination_rate=0.02
# experiment settings
runs=1

### DPLAN Settings
settings={}
settings["hidden_layer"]=20 # l
settings["memory_size"]=100000 # M
settings["warmup_steps"]=10000
settings["episodes"]=10
settings["steps_per_episode"]=2000
settings["epsilon_max"]=1
settings["epsilon_min"]=0.1
settings["epsilon_course"]=10000
settings["minibatch_size"]=32
settings["discount_factor"]=0.99 # gamma
settings["learning_rate"]=0.00025
settings["minsquared_gradient"]=0.01
settings["gradient_momentum"]=0.95
settings["penulti_update"]=2000 # N
settings["target_update"]=10000 # K

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
