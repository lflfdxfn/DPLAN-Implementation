import os
import numpy as np
import pandas as pd

from DPLAN import DPLAN
from ADEnv import ADEnv
from utils import writeResults

### Basic Settings
# data path settings
data_path="/data/zhicao/UnknownAD"
data_folders=["NB15_unknown1"]
data_subsets={"NB15_unknown1":["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance"]}
testdata_subset="test_for_all.csv" # test data is the same for subsets of the same class
# scenario settings
num_knowns=60
contamination_rate=0.02
# experiment settings
runs=1
model_path="./model"
result_path="./results"
result_file="results.csv"

### Anomaly Detection Environment Settings
size_sampling_Du=1000
prob_au=0.5
label_normal=0
label_anomaly=1

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
    testdata_path=os.path.join(data_path,data_f,testdata_subset)
    test_table=pd.read_csv(testdata_path)
    test_dataset=test_table.values

    for subset in subsets:
        # location of unknwon datasets
        unknown_dataname="{}_{}_{}.csv".format(subset,contamination_rate,num_knowns)
        undata_path=os.path.join(data_path,data_f,unknown_dataname)
        # get unknown dataset
        table=pd.read_csv(undata_path)
        undataset=table.values

        print()
        rocs=[]
        prs=[]
        # run experiment
        for i in range(runs):
            env=ADEnv(dataset=undataset,
                      sampling_Du=size_sampling_Du,
                      prob_au=prob_au,
                      label_normal=label_normal,
                      label_anomaly=label_anomaly)
            roc,pr,agent=DPLAN(env=env,settings=settings,testdata=test_dataset)

            # write weights
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            agent.model.save_weights("models/dqn_{}_{}.h5f".format(subset,i),overwrite=True)
            print("{} Run {}: AUC-ROC: {:.4f}, AUC-PR: {:.4f}".format(subset,i,roc,pr))

            rocs.append(roc)
            prs.append(pr)

        # write results
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        writeResults(name, rocs, prs, os.path.join(result_path,result_file))