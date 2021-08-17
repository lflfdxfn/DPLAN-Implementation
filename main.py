import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
tf.device("/cpu:0")

from DPLAN import DPLAN
from ADEnv import ADEnv
from utils import writeResults
from sklearn.metrics import roc_auc_score, average_precision_score

### Basic Settings
# data path settings
data_path="D:\\Datasets\\UAD"
data_folders=["NB15_unknown1"]
# data_subsets={"NB15_unknown1":["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance"]}
data_subsets={"NB15_unknown1":["Fuzzers","Generic","Reconnaissance"]}
testdata_subset="test_for_all.csv" # test data is the same for subsets of the same class
# scenario settings
num_knowns=60
contamination_rate=0.02
# experiment settings
runs=10
model_path="./model"
result_path="./results"
result_file="results.csv"
Train=True
Test=True

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
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)
for data_f in data_folders:
    # different unknown datasets for each dataset
    subsets=data_subsets[data_f]
    testdata_path=os.path.join(data_path,data_f,testdata_subset)
    test_table=pd.read_csv(testdata_path)
    test_dataset=test_table.values

    for subset in subsets:
        np.random.seed(42)
        tf.random.set_seed(42)
        # location of unknwon datasets
        data_name="{}_{}_{}".format(subset,contamination_rate,num_knowns)
        unknown_dataname=data_name+".csv"
        undata_path=os.path.join(data_path,data_f,unknown_dataname)
        # get unknown dataset
        table=pd.read_csv(undata_path)
        undataset=table.values

        print()
        rocs=[]
        prs=[]
        train_times=[]
        test_times=[]
        # run experiment
        for i in range(runs):
            print("#######################################################################")
            print("Dataset: {}".format(subset))
            print("Run: {}".format(i))

            weights_file=os.path.join(model_path,"{}_{}_{}_weights.h4f".format(subset,i,data_name))
            # initialize environment and agent
            tf.compat.v1.reset_default_graph()
            env=ADEnv(dataset=undataset,
                      sampling_Du=size_sampling_Du,
                      prob_au=prob_au,
                      label_normal=label_normal,
                      label_anomaly=label_anomaly,
                      name=data_name)
            model=DPLAN(env=env,
                        settings=settings)

            # train the agent
            train_time=0
            if Train:
                # train DPLAN
                train_start=time.time()
                model.fit(weights_file=weights_file)
                train_end=time.time()
                train_time=train_end-train_start
                print("Train time: {}/s".format(train_time))

            # test the agent
            test_time=0
            if Test:
                test_X, test_y=test_dataset[:,:-1], test_dataset[:,-1]
                model.load_weights(weights_file)
                # test DPLAN
                test_start=time.time()
                pred_y=model.predict(test_X)
                test_end=time.time()
                test_time=test_end-test_start
                print("Test time: {}/s".format(test_time))

                roc = roc_auc_score(test_y, pred_y)
                pr = average_precision_score(test_y, pred_y)
                print("{} Run {}: AUC-ROC: {:.4f}, AUC-PR: {:.4f}, train_time: {:.2f}, test_time: {:.2f}".format(subset,
                                                                                                                 i,
                                                                                                                 roc,
                                                                                                                 pr,
                                                                                                                 train_time,
                                                                                                                 test_time))

                rocs.append(roc)
                prs.append(pr)
                train_times.append(train_time)
                test_times.append(test_time)

        if Test:
            # write results
            writeResults(subset, rocs, prs, train_times, test_times, os.path.join(result_path,result_file))