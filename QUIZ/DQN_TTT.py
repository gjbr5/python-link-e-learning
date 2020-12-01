import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# %%

import random
from collections import deque
from ttt_env import TicTacToe

map_N = 3

env = TicTacToe()

input_size = map_N * map_N
output_size = map_N * map_N

# 학습 매캐변수를 설정한다.
learning_rate = .0005
discount_reward = .99
num_episodes = 100000
max_step = 100
epsilon_decay = 0.9999
initial_epsilon = 1.0
min_epsilon = 0.01
batch_size = 32
replay_counter = 0
draw_interval = 500


# Q-Network 생성
class Agent:
    def __init__(self):
        self.input = tf.keras.Input(shape=[input_size, ], dtype='float32', name='input')
        self.output = tf.keras.layers.Dense(64, kernel_initializer='he_normal')(self.input)
        self.output = tf.keras.layers.LeakyReLU(alpha=0.05)(self.output)
        self.output = tf.keras.layers.Dense(64, kernel_initializer='he_normal')(self.output)
        self.output = tf.keras.layers.LeakyReLU(alpha=0.05)(self.output)
        self.output = tf.keras.layers.Dense(output_size, activation='linear')(self.output)

        self.Q_Network_model = tf.keras.Model(self.input, self.output)
        # self.Q_Network_model.summary()

        self.Q_Network_model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate)
        )

        self.target_output = tf.keras.layers.Dense(64, kernel_initializer='he_normal')(self.input)
        self.target_output = tf.keras.layers.LeakyReLU(alpha=0.05)(self.target_output)
        self.target_output = tf.keras.layers.Dense(64, kernel_initializer='he_normal')(self.target_output)
        self.target_output = tf.keras.layers.LeakyReLU(alpha=0.05)(self.target_output)
        self.target_output = tf.keras.layers.Dense(output_size, activation='linear')(self.target_output)

        self.target_Q_Network_model = tf.keras.Model(self.input, self.target_output)
        # self.target_Q_Network_model.summary()

        self.target_Q_Network_model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate)
        )

        self.epsilon = initial_epsilon

    def forward(self, observation):
        return self.Q_Network_model.predict(observation)

    def get_action(self, observation):
        if self.epsilon > min_epsilon:
            self.epsilon *= epsilon_decay
        if np.random.rand(1) < self.epsilon:
            self.action = random.randint(0, output_size - 1)
        else:
            self.action = np.argmax(self.forward(observation))
        return self.action

    def update_weights_target_model(self):
        self.target_Q_Network_model.set_weights(
            self.Q_Network_model.get_weights()
        )

    def update_from_memory(self, memory):
        global replay_counter
        observation_batch = np.ndarray(shape=[batch_size, input_size])
        y_batch = np.ndarray(shape=[batch_size, output_size])

        minibatch = random.sample(memory, batch_size)
        i = 0
        for sample in minibatch:
            observation, action, reward, next_observation, done = sample  # unpacking

            # policy prediction for a given state
            Q_values = self.Q_Network_model.predict(observation)
            if done:
                Q_values[0, action] = reward
            else:
                Q_values_next = self.target_Q_Network_model.predict(next_observation)
                Q_values[0, action] = reward + discount_reward * np.max(Q_values_next)
            observation_batch[i] = observation
            y_batch[i] = Q_values
            i += 1

        # DQN 알고리즘으로 학습
        self.Q_Network_model.fit(
            x=np.array(observation_batch),
            y=np.array(y_batch),
            batch_size=batch_size,
            epochs=2,
            verbose=0
        )
        if replay_counter % 100 == 0:
            self.update_weights_target_model()
        replay_counter += 1

    def update_from_transition(self, transition):
        global replay_counter
        observation_batch = np.ndarray(shape=[1, input_size])
        y_batch = np.ndarray(shape=[1, output_size])

        observation, action, reward, next_observation, done = transition  # unpacking

        # policy prediction for a given state
        Q_values = self.Q_Network_model.predict(observation)
        if done:
            Q_values[0, action] = reward
        else:
            Q_values_next = self.target_Q_Network_model.predict(next_observation)
            Q_values[0, action] = reward + discount_reward * np.max(Q_values_next)
        observation_batch[0] = observation
        y_batch[0] = Q_values

        # DQN 알고리즘으로 학습
        self.Q_Network_model.fit(
            x=np.array(observation_batch),
            y=np.array(y_batch),
            batch_size=batch_size,
            epochs=2,
            verbose=0
        )
        if replay_counter % 10 == 0:
            self.update_weights_target_model()
        replay_counter += 1


def print_current_map(observation):
    for h in range(map_N):
        for w in range(map_N):
            print(observation[h][w], end=' ')
        print('\n', end='')
    print()


def find_index(data, target):
    res = []
    lis = data
    while True:
        try:
            res.append(lis.index(target) + (res[-1]+1 if len(res)!=0 else 0))
            lis = data[res[-1]+1:]
        except:
            break
    return res

def get_moving_average_list(_input, ma_window_size):
    ma_reward_all_X = []
    ma_win = deque(maxlen=ma_window_size)
    for r in _input:
        ma_win.append(r)
        ma_reward_all_X.append(np.mean(ma_win))
    return ma_reward_all_X

if __name__ == '__main__':
    play = True
    agent = Agent()
    if play:
        agent.Q_Network_model = tf.keras.models.load_model('saved_model_X_last')

    reward_all_X = []

    for current_episode_num in range(num_episodes):
        # 환경을 리셋하고 첫 번째 새로운 관찰(observation)을 얻는다.
        observation = env.reset()
        done = False

        # Q 테이블 학습 알고리즘
        transitions = []
        transition_save_step = 0
        while not done:
            # Q 테이블로부터 (노이즈와 함께) 그리디하게 액션을 선택
            if play:
                observation = np.reshape(observation, (1, input_size))
                agent.epsilon = 0.0
                action = agent.get_action(observation)
                next_observation, reward, done = env.step(action)
                print_current_map(next_observation)
                if done:
                    print("결과: ", reward)
                    break
                else:
                    action = int(input())
                    # 환경으로부터 새로운 상태와 보상을 얻는다.
                    next_observation, reward, done = env.step(action)
                    if done:
                        print_current_map(next_observation)
                        print("결과: ", reward)
                        break
                    else:
                        observation = next_observation
            else:
                observation = np.reshape(observation, (1, input_size))
                action = agent.get_action(observation)  # agent
                next_observation, reward, done = env.step(action)
                base_reward = reward
                reward = reward['X']
                next_observation = np.reshape(next_observation, (1, input_size))

                if done:
                    transition = [observation, action, reward, next_observation, done]
                    agent.update_from_transition(transition)
                    reward_all_X.append(reward)

                    if current_episode_num % draw_interval == 0:
                        ma_reward_all_X = get_moving_average_list(reward_all_X, 100)
                        plt.plot(ma_reward_all_X)
                        plt.savefig('reward_all_X.png', dpi=300)
                        # agent.Q_Network_model.save('saved_model_X')
                        print("episode: {0}, epsilon: {1}".format(current_episode_num, agent.epsilon))
                        print(base_reward)

                    break
                else:
                    # random agent
                    random_action = random.choice(find_index(list(next_observation[0]), 0))
                    # 환경으로부터 새로운 상태와 보상을 얻는다.
                    next_observation, reward, done = env.step(random_action)
                    base_reward = reward
                    reward = reward['X']
                    next_observation = np.reshape(next_observation, (1, input_size))

                    if done:
                        transition = [observation, action, reward, next_observation, done]
                        agent.update_from_transition(transition)
                        reward_all_X.append(reward)

                        if current_episode_num % draw_interval == 0:
                            ma_reward_all_X = get_moving_average_list(reward_all_X, 100)
                            plt.plot(ma_reward_all_X)
                            plt.savefig('reward_all_X.png', dpi=300)
                            # agent.Q_Network_model.save('saved_model_X')
                            print("episode: {0}, epsilon: {1}".format(current_episode_num, agent.epsilon))
                            print(base_reward)
                    else:
                        transition = [observation, action, reward, next_observation, done]
                        agent.update_from_transition(transition)

                    observation = next_observation

    if not play:
        agent.Q_Network_model.save('saved_model_X')

        ma_reward_all_X = get_moving_average_list(reward_all_X, 100)

        plt.plot(ma_reward_all_X)
        plt.savefig('reward_all_X.png', dpi=300)
        plt.show()
