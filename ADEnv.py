import gym
import time
import numpy as np

from utils import penulti_output
from gym import spaces

class ADEnv(gym.Env):
    """
    Customized environment for anomaly detection
    """
    def __init__(self,dataset: np.ndarray,sampling_Du=1000,prob_au=0.5,label_normal=0,label_anomaly=1, name="default"):
        """
        Initialize anomaly environment for DPLAN algorithm.
        :param dataset: Input dataset in the form of 2-D array. The Last column is the label.
        :param sampling_Du: Number of sampling on D_u for the generator g_u
        :param prob_au: Probability of performing g_a.
        :param label_normal: label of normal instances
        :param label_anomaly: label of anomaly instances
        """
        super().__init__()
        self.name=name

        # hyperparameters:
        self.num_S=sampling_Du
        self.normal=label_normal
        self.anomaly=label_anomaly
        self.prob=prob_au

        # Dataset infos: D_a and D_u
        self.m,self.n=dataset.shape
        self.n_feature=self.n-1
        self.n_samples=self.m
        self.x=dataset[:,:self.n_feature]
        self.y=dataset[:,self.n_feature]
        self.dataset=dataset
        self.index_u=np.where(self.y==self.normal)[0]
        self.index_a=np.where(self.y==self.anomaly)[0]

        # observation space:
        self.observation_space=spaces.Discrete(self.m)

        # action space: 0 or 1
        self.action_space=spaces.Discrete(2)

        # initial state
        self.counts=None
        self.state=None
        self.DQN=None

    def generater_a(self, *args, **kwargs):
        # sampling function for D_a
        index=np.random.choice(self.index_a)

        return index

    def generate_u(self,action,s_t):
        # sampling function for D_u
        S=np.random.choice(self.index_u,self.num_S)
        # calculate distance in the space of last hidden layer of DQN
        all_x=self.x[np.append(S,s_t)]

        all_dqn_s=penulti_output(all_x,self.DQN)
        dqn_s=all_dqn_s[:-1]
        dqn_st=all_dqn_s[-1]

        dist=np.linalg.norm(dqn_s-dqn_st,axis=1)

        if action==1:
            loc=np.argmin(dist)
        elif action==0:
            loc=np.argmax(dist)
        index=S[loc]

        return index

    def reward_h(self,action,s_t):
        # Anomaly-biased External Handcrafted Reward Function h
        if (action==1) & (s_t in self.index_a):
            return 1
        elif (action==0) & (s_t in self.index_u):
            return 0

        return -1

    def step(self,action):
        # store former state
        s_t=self.state
        # choose generator
        g=np.random.choice([self.generater_a, self.generate_u])
        s_tp1=g(action,s_t)

        # change to the next state
        self.state=s_tp1
        self.counts+=1

        # calculate the reward
        reward=self.reward_h(action,s_t)

        # done: whether terminal or not
        done=False

        # info
        info={"State t":s_t, "Action t": action, "State t+1":s_tp1}

        return self.state, reward, done, info

    def reset(self):
        # reset the status of environment
        self.counts=0
        # the first observation is uniformly sampled from the D_u
        self.state=np.random.choice(self.index_u)

        return self.state