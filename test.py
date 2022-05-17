from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp


def get_SP_tickers() -> list[str]:
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find(id="constituents")
    return [
        row.find('a').text
    for row in table.tbody.find_all("tr")][1:]


class Critic(keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(4, activation='relu')
        self.v = layers.Dense(1, activation=None)


class Actor(keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(4, activation='relu')
        self.a = layers.Dense(3, activation=None)
    

class Agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = Actor()
        self.critic = Critic()

          
    def act(self,state):
        prob = self.actor(np.array([state]))
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
            c_loss = 0.5*np.square(np.subtract(discnt_rewards, v)).mean()
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
    states = np.array(states, dtype=np.float32)
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
    state = None # TODO reset the env here
    total_reward = 0
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    actions = []

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = 0 # TODO return these variables by interacting with the environment
        rewards.append(reward)
        states.append(state)
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