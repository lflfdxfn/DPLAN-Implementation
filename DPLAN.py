import tensorflow
import numpy as np

from rl.core import Processor
from rl.util import clone_model
from rl.memory import Memory, SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.callbacks import Callback, FileLogger, ModelIntervalCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import penulti_output
from ADEnv import ADEnv


def QNetwork(input_shape,hidden_unit=20):
    x_input=Input(shape=(1,input_shape))
    flatten_x=Flatten(input_shape=(1,input_shape))(x_input)
    latent_x=Dense(hidden_unit,activation='relu',kernel_regularizer=regularizers.l2(0.01))(flatten_x)
    x=Dense(2,activation='linear')(latent_x)

    return Model(x_input, x)

def DQN_iforest(x, model: Model):
    # iforest function on the penuli-layer space of DQN

    # get the output of penulti-layer
    latent_x=penulti_output(x,model)
    # calculate anomaly scores in the latent space
    iforest=IsolationForest().fit(latent_x)
    scores=-iforest.score_samples(latent_x)
    # scaler scores to [0,1]
    norm_scores=(scores-scores.min())/(scores.max()-scores.min())

    return norm_scores

class DPLANProcessor(Processor):
    """
    Customize the fit function of DQNAgent.
    """
    def __init__(self, env: ADEnv):
        """
        :param env: Used to get the dataset from the environment.
        """
        self.x=env.x
        self.intrinsic_reward=None

        # store the index of s_t
        self.last_observation=None

    def process_step(self, observation, reward, done, info):
        # note that process_step runs after the step of environment
        # if we only modify the process_observation function,
        # the last_observation attribute will change to s_t+1 before the intrinsic reward is added.
        last_observation=self.last_observation # stored beore changed.

        observation=self.process_observation(observation)
        reward=self.process_reward(reward,last_observation)
        info=self.process_info(info)

        return observation, reward, done, info

    def process_observation(self,observation):
        # note that the observation generated from ADEnv is the index of the point in dataset
        # convert it to a numpy array
        self.last_observation=observation

        return self.x[observation,:]

    def process_reward(self, reward_e, last_observation):
        # integrate the intrinsic reward function
        reward_i=self.intrinsic_reward[last_observation]

        return reward_e+reward_i

class DPLANCallbacks(Callback):
    def on_action_begin(self, action, logs={}):
        self.env.DQN=self.model.model

    def on_train_begin(self, logs=None):
        # calculate the intrinsic_reward from the initialized DQN
        self.model.processor.intrinsic_reward=DQN_iforest(self.env.x, self.model.model)

    def on_episode_end(self, episode, logs={}):
        # on the end of episode, DPLAN needs to update the target DQN and the penulti-features
        # the update process of target DQN have implemented in "rl.agents.dqn.DQNAgent.backward()"
        self.model.processor.intrinsic_reward=DQN_iforest(self.env.x, self.model.model)


def DPLAN(env: ADEnv, settings: dict, testdata: np.ndarray, model_name, mode="train", *args, **kwargs):
    """
    1. Train a DPLAN model on anomaly-detection environment.
    2. Test it on the test dataset.
    3. Return the predictions.
    :param env: Environment of the anomaly detection.
    :param settings: Settings of hyperparameters in dict format.
    :param testdata: Test dataset ndarray. The last column contains the labels.
    :param model_name: Name of trained model.
    :param mode: Train or Test.
    """
    # hyperparameters
    l=settings["hidden_layer"]
    M=settings["memory_size"]
    warmup_steps=settings["warmup_steps"]
    n_episodes=settings["episodes"]
    n_steps_episode=settings["steps_per_episode"]
    max_epsilon=settings["epsilon_max"]
    min_epsilon=settings["epsilon_min"]
    greedy_course=settings["epsilon_course"]
    minibatch_size=settings["minibatch_size"]
    gamma=settings["discount_factor"]
    lr=settings["learning_rate"]
    min_grad=settings["minsquared_gradient"]
    grad_momentum=settings["gradient_momentum"]
    N=settings["penulti_update"] # hyper-parameter not used
    K=settings["target_update"]

    # initialize DQN Agent
    input_shape=env.n_feature
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
    processor=DPLANProcessor(env)
    agent=DQNAgent(model=model,
                   policy=policy,
                   nb_actions=n_actions,
                   memory=memory,
                   processor=processor,
                   gamma=gamma,
                   batch_size=minibatch_size,
                   nb_steps_warmup=warmup_steps,
                   train_interval=1,#update frequency
                   target_model_update=K)
    # optimizer=RMSprop(learning_rate=lr, momentum=grad_momentum,epsilon=min_grad)
    optimizer=Adam(learning_rate=lr, epsilon=min_grad)
    agent.compile(optimizer=optimizer)
    # initialize target DQN with weight=0
    weights=agent.model.get_weights()
    for weight in weights:
        weight[:]=0
    agent.target_model.set_weights(weights)

    weights_filename = "{}_{}_weights.h5f".format(model_name,env.name)
    if mode=="train":
        # train DPLAN
        callbacks=DPLANCallbacks()
        agent.fit(env=env,
                  nb_steps=warmup_steps+n_episodes*n_steps_episode,
                  action_repetition=1,
                  callbacks=[callbacks],
                  nb_max_episode_steps=n_steps_episode)
        agent.save_weights(weights_filename,overwrite=True)
    elif mode=="test":
        agent.load_weights(weights_filename)
        # test DPLAN
        x,y=testdata[:,:-1], testdata[:,-1]
        q_values=agent.model.predict(x[:,np.newaxis,:])
        scores=q_values[:,1]
        # scores=np.argmax(q_values,axis=1)
        roc=roc_auc_score(y,scores)
        pr=average_precision_score(y,scores)

        return roc, pr