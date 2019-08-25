import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from LunarLandarContinuous import LunarLanderContinuous


class ContinuousDeepQNet:

    def __init__(self, observation_space, action_space, action_max, action_min, environment, exploration_rate=1.0,
                 exploration_rate_decay=0.995, memory_maxlen=10000, learning_rate=0.001, discount_rate=0.95,
                 verbose=False):
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.discount_rate = discount_rate

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_max = action_max
        self.action_min = action_min
        self.memory = deque(maxlen=memory_maxlen)

        self.env = environment

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.online_model = Sequential()
        self.online_model.add(Dense(24, input_shape=(observation_space,), activation='relu'))
        self.online_model.add(Dense(12, activation='relu'))
        self.online_model.add(Dense(action_space, activation='linear'))

        self.live_model = Sequential()
        self.live_model.add(Dense(24, input_shape=(observation_space,)))
        self.live_model.add(Dense(12, activation='relu'))
        self.live_model.add(Dense(action_space, activation='linear'))

        self.verbose = verbose

        if self.verbose:
            self.online_model.summary()

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def experience_replay(self, minibatch_size):
        if len(self.memory) < minibatch_size:
            return np.asarray(list(self.memory))
        return np.asarray(random.sample(self.memory, minibatch_size))

    def act(self, state):
        if np.random.random() < self.exploration_rate:
            action = np.random.uniform(self.action_min, self.action_max, self.action_space)
        else:
            action = self.live_model.predict(state)[0]
        print(action)
        return action

    def train(self, episodes, minibatch_size=16, render=False):
        for episode in range(episodes):
            state = np.reshape(self.env.reset(), (1, self.observation_space))
            terminal = False
            step = 1

            while not terminal:
                if render:
                    self.env.render()

                action = self.act(state)
                next_state, reward, terminal, info = self.env.step(action)
                self.remember(state, action, reward, next_state, terminal)

                replays = self.experience_replay(1)

                with tf.GradientTape() as tape:
                    replay = replays[0]
                    q = tf.constant(self.live_model(replay[0]))
                    if replay[4]:
                        y = tf.constant(replay[2], shape=(1, self.action_space), dtype=tf.float32)
                    else:
                        y = tf.add(tf.constant(replay[2], dtype=tf.float32),
                                   tf.scalar_mul(tf.constant(self.discount_rate, dtype=tf.float32),
                                                 self.online_model(replay[0], training=True)))

                    loss = tf.losses.mean_squared_error(y, q)

                grads = tape.gradient(loss, self.online_model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                clipped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
                self.optimizer.apply_gradients(zip(clipped_grads, self.online_model.trainable_variables), global_step=tf.train.get_or_create_global_step())

                self.exploration_rate *= self.exploration_rate_decay

                if step % 10 == 0:
                    self.live_model.set_weights(self.online_model.get_weights())
                    step = 0
                else:
                    step += 1

                if terminal and self.verbose:
                    pass
                    print('Run:', episode, ', reward:', reward)

    def _loss(self, y, q):
        return tf.reduce_mean(tf.square(tf.subtract(y, q)), axis=1)


def testfunc():
    tf.enable_eager_execution()
    env = gym.make("LunarLanderContinuous-v2")
    # env = LunarLanderContinuous()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    dqn_solver = ContinuousDeepQNet(observation_space, action_space, 1, -1, env, verbose=True, learning_rate=0.0001)
    tf.initializers.global_variables()

    dqn_solver.train(200, render=True)

testfunc()
