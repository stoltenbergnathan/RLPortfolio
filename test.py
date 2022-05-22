import enum
import sys
from typing import Any, Tuple
from matplotlib import pyplot as plt
import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp
import yfinance as yf

class Actions(enum.Enum):
    SELL = 0
    HOLD = 1
    BUY = 2

class Env():
    def __init__(self, ticker) -> None:
        self.ticker = ticker
        self.day = 0
        self.history = self.get_history(ticker)
    
    def get_history(self, ticker) -> list[float]:
        tl = yf.Ticker(ticker)
        history: pandas.DataFrame = tl.history(period = "2y", interval = "1d")
        datalist = []
        for i in history.index:
            datalist.append(history.loc[[i], ["High"]].values[0][0])
        return datalist
    
    def get_highs(self) -> list[float]:
        dl = []
        if len(self.history) - self.day < 10:
            return dl
        for index in range(self.day, self.day + 10):
            dl.append(self.history[index])
        self.day += 1
        return dl


class State():
    def __init__(self, highs) -> None:
        self.highs = highs
        self.currentPrice = highs[-1] if len(highs) != 0 else -1
    
    def preform_action(self, action: Actions, env: Env) -> Tuple[Any, float, bool]:
        new_highs = env.get_highs()
        if len(new_highs) != 0:
            change = 100 * (1 - (self.currentPrice / new_highs[-1]))
            if action == Actions.BUY.value:
                if change > 0:
                    Rreward = change
                else:
                    Rreward = -1 * change
            elif action == Actions.SELL.value:
                if change < 0:
                    Rreward = -1 * change
                else:
                    Rreward = change
            elif action == Actions.HOLD.value:
                Rreward = 0
            else:
                print(action)
                sys.exit(1)
            return State(new_highs), Rreward, False
        else:
            return State([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0, True


class Critic(keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(10, activation='relu')
        self.d1 = layers.Dense(5, activation='relu')
        self.v = layers.Dense(1, activation=None)
    

    def call(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v


class Actor(keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(10, activation='relu')
        self.a = layers.Dense(3, activation='softmax')
    

    def call(self, input_data):
        x = self.d1(input_data)
        a = self.a(x)
        return a 
    

class Agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = Actor()
        self.critic = Critic()

          
    def act(self,state):
        prob = self.actor(np.array([np.array(state)]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, td):
        probability = []
        log_probability= []
        for pb,a in zip(probs,actions):
          dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
          log_prob = dist.log_prob(a)
          prob = dist.prob(a)
          probability.append(prob)
          log_probability.append(log_prob)
        p_loss= []
        e_loss = []
        td = td.numpy()
        for pb, t, lpb in zip(probability, td, log_probability):
            t =  tf.constant(t)
            policy_loss = tf.math.multiply(lpb,t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb,lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        return loss
    

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards), ))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, td)
            c_loss = keras.metrics.mean_squared_error(discnt_rewards, v)
            grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
            grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
            self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
            self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
            return a_loss, c_loss


def preproccess(states, actions, rewards, gamma):
    discnt_rewards = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
      sum_reward = r + gamma*sum_reward
      discnt_rewards.append(sum_reward)
    discnt_rewards.reverse()
    states = np.vstack(states).astype(np.float64)
    actions = np.array(actions, dtype=np.int32)
    discnt_rewards = np.array(discnt_rewards, dtype=np.float32)
    return states, actions, discnt_rewards


tf.random.set_seed(336699)
agent = Agent()
steps = 250
ep_reward = []
total_avgr = []

for step in range(steps):
    done = False
    env = Env("AAPL")
    state = State(env.get_highs())
    total_reward = 0
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    actions = []

    while not done:
        action = agent.act(state.highs) - 1
        next_state, reward, done = state.preform_action(action, env)
        rewards.append(reward)
        states.append(state.highs)
        actions.append(action)
        state = next_state
        total_reward += reward

        if done:
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} steps is {} and avg reward is {}".format(step, total_reward, avg_reward))
            states, actions, discnt_rewards = preproccess(states, actions, rewards, agent.gamma)

            al, cl = agent.learn(states, actions, discnt_rewards)
            print(f"al{al}") 
            print(f"cl{cl}")


ep = [i  for i in range(250)]
plt.plot(ep,total_avgr,'b')
plt.title("avg reward Vs episods")
plt.xlabel("episods")
plt.ylabel("average reward per 100 episods")
plt.grid(True)
plt.show()