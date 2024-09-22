import gymnasium as gym
import numpy as np


class M2S_Env2(gym.Env):
    def __init__(self, config):
        self.action_space = gym.spaces.Discrete(4)
        self.attribute_dim = config["attribute_dim"]
        if "exhaustive" in config and config["exhaustive"]:
            self.exhaustive = True
        self.obs_dim = self.attribute_dim * 2 + 4
        self.observation_space = gym.spaces.Box(low=np.zeros(self.obs_dim, dtype='int'),
                                                high=np.ones(self.obs_dim, dtype='int'),
                                                dtype=int)
        self.task_switch = 0
        self.sample = np.zeros(self.attribute_dim, dtype='int')
        self.target = np.zeros(self.attribute_dim, dtype='int')
        self.spurious = np.zeros(self.attribute_dim, dtype='int')
        self.sample_period = config["sample_period"]
        self.response_period = config["response_period"]
        self.done = False
        self.count = 0
        self.episode_count = 0
        self.shuffled = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        self.done = False
        if self.exhaustive:
            if self.episode_count % 16 == 0:
                self.shuffled = np.arange(16)
                np.random.shuffle(self.shuffled)
            y = format(self.shuffled[self.episode_count % 16], '04b')
            # print(y)
            self.task_switch = int(y[0])
            self.sample = M2S_Env2.one_hot(int(y[1]), self.attribute_dim)
            self.target = M2S_Env2.one_hot(int(y[2]), self.attribute_dim)
            self.spurious = M2S_Env2.one_hot(int(y[3]), self.attribute_dim)
        else:
            self.task_switch = np.random.randint(0, 2, dtype='int')
            self.sample = M2S_Env2.one_hot(np.random.randint(0, self.attribute_dim, 1)[0], self.attribute_dim)
            self.target = M2S_Env2.one_hot(np.random.randint(0, self.attribute_dim, 1)[0], self.attribute_dim)
            self.spurious = M2S_Env2.one_hot(np.random.randint(0, self.attribute_dim, 1)[0], self.attribute_dim)
        self.episode_count += 1
        return np.zeros(self.obs_dim, dtype='int'), {}

    @staticmethod
    def one_hot(n, dim):
        buf = np.zeros(dim, dtype='int')
        buf[n] = 1
        return buf

    def step(self, action):
        self.count += 1
        zero2 = np.array([0, 0], dtype='int')
        if self.count <= 1:  # + 1:
            if self.task_switch == 0:
                observation = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype='int')
            else:
                observation = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype='int')
        elif self.count <= self.sample_period + 1:
            # showing the sample
            if self.task_switch == 0:
                observation = np.append(zero2, np.append(np.append(self.sample, self.spurious), zero2))
            else:
                observation = np.append(zero2, np.append(np.append(self.spurious, self.sample), zero2))
        elif self.count <= self.sample_period + 2:  # + 1:   # match delay
            # delay & prepare the target
            observation = np.zeros(self.obs_dim, dtype='int')
        elif self.count <= self.sample_period + self.response_period + 2:
            # showing the target
            if self.task_switch == 0:
                observation = np.append(zero2, np.append(np.append(self.target, self.spurious), zero2))
            else:
                observation = np.append(zero2, np.append(np.append(self.spurious, self.target), zero2))
        elif self.count <= self.sample_period + self.response_period + 3:  # answer delay
            observation = np.zeros(self.obs_dim, dtype='int')
        else:   # showing the answer
            if np.all(self.sample == self.target):
                observation = np.append(np.zeros(self.obs_dim - 2, dtype='int'), np.array([1, 0], dtype='int'))
            else:
                observation = np.append(np.zeros(self.obs_dim - 2, dtype='int'), np.array([0, 1], dtype='int'))
            self.done = True
        return observation, 0.0, self.done, False, {}

    def render(self):
        pass


def main():
    config = {"attribute_dim": 2, "sample_period": 2, "response_period": 2, "exhaustive": True}
    env = M2S_Env2(config)
    for i in range(16):
        env.reset()
        action = np.random.randint(1, 3)
        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs)
            if terminated:
                print('--')
                break


if __name__ == '__main__':
    main()
