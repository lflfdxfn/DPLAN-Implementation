from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def penulti_output(x, DQN: Model):
    inp = DQN.input
    penulti_func = K.function([inp], [DQN.layers[-2].output])
    latent_x = penulti_func(x)[0]

    return latent_x