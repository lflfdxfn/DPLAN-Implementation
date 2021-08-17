from tensorflow.keras.models import Model
import os
import numpy as np
import tensorflow.keras.backend as K

def penulti_output(x: np.ndarray, DQN: Model):
    inp = DQN.input
    penulti_func = K.function([inp], [DQN.layers[-2].output])
    latent_x = penulti_func(x)[0]

    return latent_x

def writeResults(name,rocs,prs,train_times, test_times, file_path):
    roc_mean=np.mean(rocs)
    roc_std=np.std(rocs)
    pr_mean=np.mean(prs)
    pr_std=np.std(prs)
    train_mean=np.mean(train_times)
    train_std=np.std(train_times)
    test_mean=np.mean(test_times)
    test_std=np.std(test_times)

    header=True
    if not os.path.exists(file_path):
        header=False

    with open(file_path,'a') as f:
        if not header:
            f.write("{}, {}, {}, {}, {}\n".format("Name",
                                        "AUC-ROC(mean/std)",
                                        "AUC-PR(mean/std)",
                                        "Train time/s",
                                        "Test time/s"))

        f.write("{}, {}/{}, {}/{}, {}/{}, {}/{}\n".format(name,
                                                          roc_mean, roc_std,
                                                          pr_mean, pr_std,
                                                          train_mean, train_std,
                                                          test_mean, test_std))