import numpy as np
from rl.agents.dqn import DQNAgent
from rl.callbacks import Callback
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from sklearn.ensemble import IsolationForest
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from ADEnv import ADEnv
from utils import penulti_output


class DPLAN:
    """
    DPLAN model.
    """
    def __init__(self, env: ADEnv, settings: dict):
        """
        Initialize a DPLAN model.
        :param env: Environment of the anomaly detection.
        :param settings: Settings of hyperparameters in dict format.
        """
        # basic properties
        self.env=env
        self.train_env=None
        self.settings=settings

        # hyperparameters
        self.l = settings["hidden_layer"]
        self.M = settings["memory_size"]
        self.warmup_steps = settings["warmup_steps"]
        self.n_episodes = settings["episodes"]
        self.n_steps_episode = settings["steps_per_episode"]
        self.max_epsilon = settings["epsilon_max"]
        self.min_epsilon = settings["epsilon_min"]
        self.greedy_course = settings["epsilon_course"]
        self.minibatch_size = settings["minibatch_size"]
        self.gamma = settings["discount_factor"]
        self.lr = settings["learning_rate"]
        self.min_grad = settings["minsquared_gradient"]
        self.grad_momentum = settings["gradient_momentum"]
        # hyper-parameter not used, penulti-output is updated at the end of each episode.
        self.N = settings["penulti_update"]
        self.K = settings["target_update"]

        # initialize an DQN Agent
        input_shape = env.n_feature
        n_actions = env.action_space.n
        model = QNetwork(input_shape=input_shape,
                         hidden_unit=self.l)
        policy = LinearAnnealedPolicy(inner_policy=EpsGreedyQPolicy(),
                                      attr='eps',
                                      value_max=self.max_epsilon,
                                      value_min=self.min_epsilon,
                                      value_test=0.,
                                      nb_steps=self.greedy_course)
        memory = SequentialMemory(limit=self.M,
                                  window_length=1)
        processor = DPLANProcessor(env)
        optimizer = RMSprop(learning_rate=self.lr,
                            momentum=self.grad_momentum,
                            epsilon=self.min_grad)

        agent = DQNAgent(model=model,
                         policy=policy,
                         nb_actions=n_actions,
                         memory=memory,
                         processor=processor,
                         gamma=self.gamma,
                         batch_size=self.minibatch_size,
                         nb_steps_warmup=self.warmup_steps,
                         train_interval=1,  # update frequency
                         target_model_update=self.K)
        agent.compile(optimizer=optimizer)
        # initialize target DQN with weight=0
        weights = agent.model.get_weights()
        for weight in weights:
            weight[:] = 0
        agent.target_model.set_weights(weights)

        self.agent=agent

    def fit(self, env: ADEnv=None, weights_file=None):
        # Check whether a new env is used.
        if env:
            self.train_env=env
        else:
            self.train_env=self.env

        # Train DPLAN.
        callbacks = DPLANCallbacks()
        self.agent.fit(env=self.train_env,
                       nb_steps=self.warmup_steps + self.n_episodes * self.n_steps_episode,
                       action_repetition=1,
                       callbacks=[callbacks],
                       nb_max_episode_steps=self.n_steps_episode)
        # Save weights if the weights_filename is given.
        if weights_file:
            self.agent.save_weights(weights_file, overwrite=True)

    def load_weights(self,weights_file):
        # Load weights to the DQN agent from a stored file.
        self.agent.load_weights(weights_file)

    def predict(self, X):
        # Predict current DQN agent on the test dataset.
        # Return the predicted anomaly score.
        q_values=self.agent.model.predict(X[:,np.newaxis,:])
        scores=q_values[:,1]

        return scores

    def predict_label(self, X):
        # Predict current DQN agent on the test dataset.
        # Return the predicted labels.
        q_values=self.agent.model.predict(X[:,np.newaxis,:])
        labels=np.argmax(q_values,axis=1)

        return labels


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