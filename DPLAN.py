from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from keras import regularizers
from keras import backend
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.initializers import Zeros

from ADEnv import ADEnv

class QNetwork(Model):
    """
    Network architecture with one hidden layer
    """
    def __init__(self,input_shape, hidden_unit=20):
        super().__init__()
        self.input=Input(shape=input_shape)
        self.hidden=Dense(hidden_unit,activation="relu",kernel_regularizer=regularizers.l2(0.01))
        self.output=Dense(2,activation="linear")

    def penulti_layer(self,state):
        x=self.input(state)
        x=self.hidden(x)

        # output of penultilayer
        return x.detach()

    def call(self,state):
        x=self.input(state)
        x=self.hidden(x)
        x=self.output(x)

        return x

def DPLAN(env: ADEnv, settings: dict, testdata, *args, **kwargs):
    """
    1. Train a DPLAN model on anomaly-detection environment.
    2. Test it on the test dataset.
    3. Return the predictions.
    :param env: Environment of the anomaly detection.
    :param settings: settings of hyperparameters in dict format.
    :param testdata: test dataset.
    """
    # size of dataset
    m,n=env.m,env.n
    # initialize DQN Agent
    # initialize agent's experiences
    # set optimizer
    optimizer=RMSprop(learning_rate=0.00025, clipnorm=1.,momentum=0.95)

