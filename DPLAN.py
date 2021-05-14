from rl.core import Processor
from rl.memory import Memory, SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop

from ADEnv import ADEnv

class QNetwork(Model):
    """
    Network architecture with one hidden layer.
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

class DPLANProcessor(Processor):
    """
    Customize the fit function of DQNAgent.
    """
    def process_reward(self, reward):
        # integrate the intrinsic reward function
        pass

def DPLAN(env: ADEnv, settings: dict, testdata, *args, **kwargs):
    """
    1. Train a DPLAN model on anomaly-detection environment.
    2. Test it on the test dataset.
    3. Return the predictions.
    :param env: Environment of the anomaly detection.
    :param settings: Settings of hyperparameters in dict format.
    :param testdata: Test dataset.
    """
    # hyperparameters
    l=settings["hidden_layer"]
    M=settings["memory_size"]
    warmup_steps=settings["warmup_steps"]
    n_episodes=settings["episodes"]
    n_steps=settings["steps_per_episode"]
    max_epsilon=settings["epsilon_max"]
    min_epsilon=settings["epsilon_min"]
    greedy_course=settings["epsilon_course"]
    minibatch_size=settings["minibatch_size"]
    gamma=settings["discount_factor"]
    lr=settings["learning_rate"]
    min_grad=settings["minsquared_gradient"]
    grad_momentum=settings["gradient_momentum"]
    N=settings["penulti_update"]
    K=settings["target_update"]

    # initialize DQN Agent
    input_shape=env.n
    n_actions=env.action_space.n
    model=QNetwork(input_shape=input_shape,
                   hidden_unit=l)
    policy=LinearAnnealedPolicy(inner_policy=EpsGreedyQPolicy(),
                                attr='eps',
                                value_max=max_epsilon,
                                value_min=min_epsilon,
                                value_test=0.,
                                nb_steps=greedy_course)
    memory=SequentialMemory(limit=M,
                            window_length=1)
    processor=DPLANProcessor()
    agent=DQNAgent(model=model,
                   policy=policy,
                   nb_actions=n_actions,
                   memory=memory,
                   processor=processor,
                   gamma=gamma,
                   batch_size=minibatch_size,
                   nb_steps_warmup=warmup_steps,
                   target_model_update=K,
                   )
    optimizer=RMSprop(learning_rate=lr, clipnorm=1.,momentum=grad_momentum)
    agent.compile(optimizer=optimizer)

    # train DPLAN
    for episode in range(n_episodes):
        pass