import gym
from gym import spaces

class ADEnv(gym.Env):
    """
    Customized environment for anomaly detection
    """
    def __init__(self,dataset,label_normal=0,label_anomaly=1):
        super().__init__()

        # split the dataset into D_a and D_u
        self.dataset=dataset
        m,n=dataset.shape
        x=dataset[:,:n-1]
        y=dataset[:,n-1]
        self.dataset_u=x[:,y==label_normal]
        self.dataset_a=x[:,y==label_anomaly]

        # observation space:


        # action space: 0 or 1
        self.action_space=spaces.Discrete(2)

    def step(self,action):
        # make sure action is legal
        assert self.action_space.contains(action), "Action {} (%s) is invalid".format(action,type(action))

