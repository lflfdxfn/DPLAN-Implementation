from rl.agents.dqn import DQNAgent
from keras import regularizers
from keras import backend
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.initializers import Zeros

from ADEnv import ADEnv

class DQN(Model):
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

        return x

    def call(self,state):
        x=self.input(state)
        x=self.hidden(x)
        x=self.output(x)

        return x

class DQN_target(Model):
    """
    Network architecture with one hidden layer
    """
    def __init__(self,input_shape, hidden_unit=20):
        super().__init__()
        self.input=Input(shape=input_shape)
        self.hidden=Dense(hidden_unit,activation="relu",
                          kernel_regularizer=regularizers.l2(0.01),
                          kernel_initializer=Zeros(),
                          bias_initializer=Zeros())
        self.output=Dense(2,activation="linear",
                          kernel_initializer=Zeros(),
                          bias_initializer=Zeros())

    def penulti_layer(self,state):
        x=self.input(state)
        x=self.hidden(x)

        return x

    def call(self,state):
        x=self.input(state)
        x=self.hidden(x)
        x=self.output(x)

        return x

def experiment(env: ADEnv,epoch=10,steps=2000,warm_up=10000,K=10000):
    """
    :param env: Environment of the anomaly detection
    :param epoch: training episodes
    :param steps: training steps per episode
    :param warm_up: warm-up steps
    :param K: target network in DQN to update
    :return:
    """
    # size of dataset
    m,n=env.m,env.n
    # initialize two DQN model: Q-values and target Q-values
    Q_nn=DQN(n-1)
    Q_target=DQN_target(n-1)

    optimizer=RMSprop(learning_rate=0.00025, clipnorm=1.)