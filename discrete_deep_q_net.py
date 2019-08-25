import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

import time

from gym.envs.classic_control import CartPoleEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from LunarLandarContinuous import LunarLanderContinuous, LunarLander


class DeepQNet:

    def __init__(self, environment, exploration_rate=1.0, exploration_rate_decay=0.995, memory_maxlen=10000,
                 learning_rate=0.001, discount_rate=0.95, verbose=False):
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.discount_rate = discount_rate

        self.env = environment

        self.cartpole = False

        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.memory = deque(maxlen=memory_maxlen)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.observation_space,)))
        self.model.add(Dense(12, activation='relu'))
        self.model.add(Dense(self.action_space, activation='linear'))
        self.model.compile(loss='mse', optimizer=self.optimizer)

        self.verbose = verbose

        if self.verbose:
            self.model.summary()

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def experience_replay(self, minibatch_size):
        if len(self.memory) < minibatch_size:
            return np.asarray(list(self.memory))
        return np.asarray(random.sample(self.memory, minibatch_size))

    def act(self, state):
        if np.random.random() < self.exploration_rate:
            if self.verbose:
                print('Exploring')
            return random.randrange(self.action_space)

        if self.verbose:
            print('Acting')
        q_values = self.model.predict(state)
        action = np.argmax(q_values[0])
        # print('Action:', action)
        return action

    def train(self, episodes, minibatch_size=16, render=False):
        runs = []
        rewards = []
        times = []
        for episode in range(episodes):
            state = np.reshape(self.env.reset(), (1, self.observation_space))
            terminal = False
            step = 0
            total_reward = 0
            start_time = time.time()

            while not terminal:
                if render:
                    self.env.render()

                action = self.act(state)
                next_state, reward, terminal, info = self.env.step(action)
                if self.cartpole:
                    reward = reward if not terminal else -reward  # only for cartpole
                next_state = np.reshape(next_state, (1, self.observation_space))
                self.remember(state, action, reward, next_state, terminal)
                state = next_state

                replays = self.experience_replay(minibatch_size)

                if not terminal:
                    self._train_step(replays)

                if self.exploration_rate > 0.05 / self.exploration_rate_decay:
                    self.exploration_rate *= self.exploration_rate_decay

                step += 1
                total_reward += reward

                if terminal:
                    total_time = time.time() - start_time
                    runs.append(episode)
                    rewards.append(total_reward)
                    times.append(total_time)
                    print('Run:', episode, ', reward:', reward, ', total reward', total_reward, ', time:', total_time)

        return runs, rewards, times

    def run(self, episodes=1000, render=True):
        runs = []
        rewards = []
        times = []
        for episode in range(episodes):
            state = np.reshape(self.env.reset(), (1, self.observation_space))
            terminal = False
            step = 0
            total_reward = 0
            start_time = time.time()

            while not terminal:
                if render:
                    self.env.render()

                action = self.act(state)
                next_state, reward, terminal, info = self.env.step(action)
                if self.cartpole:
                    reward = reward if not terminal else -reward  # only for cartpole
                next_state = np.reshape(next_state, (1, self.observation_space))
                state = next_state

                if self.exploration_rate > 0.05 / self.exploration_rate_decay:
                    self.exploration_rate *= self.exploration_rate_decay

                step += 1
                total_reward += reward

                if terminal:
                    total_time = time.time() - start_time
                    runs.append(episode)
                    rewards.append(total_reward)
                    times.append(total_time)
                    print('Run:', episode, ', reward:', reward, ', total reward', total_reward, ', time:', total_time)

        return runs, rewards, times

    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename)

    def _train_step(self, replays):
        for replay in replays:
            (state, action, reward, next_state, terminal) = replay
            y_action = reward
            if not terminal:
                y_action = reward + self.discount_rate * np.amax(self.model.predict(next_state)[0])

            y = self.model.predict(state)
            # print('unmodified:', y, terminal)
            y[0][action] = y_action
            # print('modified:', y)

            self.model.fit(state, y, verbose=0)


# def testfunc():
#     from matplotlib import pyplot as plt
#     tf.enable_eager_execution()
#     env = gym.make("CartPole-v1")
#     # env = LunarLander()
#     dqn_solver = DeepQNet(env, verbose=False, learning_rate=0.001, exploration_rate_decay=0.995)
#     dqn_solver.cartpole = True
#     tf.initializers.global_variables()
#
#     runs, rewards, times = dqn_solver.train(100, render=True)
#
#     plt.plot(runs, rewards, label='Rewards')
#     plt.plot(runs, times, label='Times')
#     plt.legend()
#     plt.show()
#
# testfunc()
