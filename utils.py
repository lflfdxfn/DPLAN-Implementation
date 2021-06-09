from tensorflow.keras.models import Model
import os
import time
import numpy as np
import tensorflow.keras.backend as K

def penulti_output(x: np.ndarray, DQN: Model):
    inp = DQN.input
    penulti_func = K.function([inp], [DQN.layers[-2].output])
    latent_x = penulti_func(x)[0]

    return latent_x

def writeResults(name,rocs,prs,file_path):
    roc_mean=np.mean(rocs)
    roc_std=np.std(rocs)
    pr_mean=np.mean(prs)
    pr_std=np.std(prs)

    header=True
    if not os.path.exists(file_path):
        header=False
    with open(file_path,'a') as f:
        if header==False:
            f.write("{}, {}, {}".format("name", "AUC-ROC(mean/std)", "AUC-PR(mean/std)\n"))

        f.write("{}, {}/{}, {}/{}\n".format(name, roc_mean, roc_std, pr_mean, pr_std))