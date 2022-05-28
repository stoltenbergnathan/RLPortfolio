from enum import Enum
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataFunctions import *
from os.path import exists

LOOKBACK = 10
NUM_INPUTS = LOOKBACK + 1
ACTOR_HIDDEN = [15, 15, 15, 15, 15, 15]
CRITIC_HIDDEN = [15, 15, 15, 15, 15, 15]
NUM_ACTIONS = 2
LEARNING_RATE = 0.00001


class Actions(Enum):
    INCREASE = 0
    DECREASE = 1


class State:
    def __init__(self, hist_diff: list[float], rsi: float) -> None:
        self.hist_diff = hist_diff
        self.rsi = rsi
    
    def state(self):
        tmp = self.hist_diff.copy()
        tmp.append(self.rsi)
        return tmp


class Env:
    def __init__(self, p_hist) -> None:
        self.history = get_ticker_history(p_hist)
        self.max = LOOKBACK

    def do_something(self, action: int):
        if len(self.history) - self.max < LOOKBACK:
            return State([0.0] * LOOKBACK, 0.0), 0.0, True
        else:
            hist_diff=get_diff_hist(self.history, self.max - LOOKBACK, self.max)
            new_state = State(
                hist_diff=hist_diff,
                rsi=get_RSI(hist_diff)
            )
            self.max += 1
            new_change = new_state.hist_diff[-1]
            if action == Actions.DECREASE.value and new_change < 0.0:
                reward = abs(new_change)
            elif action == Actions.INCREASE.value and new_change > 0.0:
                reward = new_change
            else:
                reward = -abs(new_change)
            return new_state, reward, False

      
    def get_init_state(self) -> State:
        diff_hist = get_diff_hist(self.history, 0, self.max)
        rsi = get_RSI(diff_hist)
        state = State(diff_hist, rsi)
        self.max += 1
        return state


    def expected_action(self):
        if len(self.history) - self.max < LOOKBACK:
            return State([0.0] * LOOKBACK, 0.0), 0.0, True
        else:
            hist_diff=get_diff_hist(self.history, self.max - LOOKBACK, self.max)
            new_state = State(
                hist_diff=hist_diff,
                rsi=get_RSI(hist_diff)
            )
            self.max += 1
            new_change = new_state.hist_diff[-1]
            if new_change > 0:
                e_action = Actions.INCREASE.value
            else:
                e_action = Actions.DECREASE.value
            return e_action, new_state, False


class Actor(tf.keras.Model):
    def __init__(self, num_inputs, num_actions, hidden_layers) -> None:
        super().__init__()
        self.hidden_layers = hidden_layers
        self.inpu = tf.keras.layers.Dense(num_inputs, activation='relu')
        self.dense_layers = [tf.keras.layers.Dense(nodes) for nodes in hidden_layers]
        self.actions = tf.keras.layers.Dense(num_actions, activation = 'softmax')
    
    def call(self, input):
        tmp = self.inpu(input)
        for layer in self.dense_layers:
            tmp = layer(tmp)
        actions = self.actions(tmp)
        return actions


class Critic(tf.keras.Model):
    def __init__(self, num_inputs, hidden_layers) -> None:
        super().__init__()
        self.hidden_layers = hidden_layers
        self.inpu = tf.keras.layers.Dense(num_inputs, activation='relu')
        self.dense_layers = [tf.keras.layers.Dense(nodes) for nodes in hidden_layers]
        self.value = tf.keras.layers.Dense(1, activation = None)
    
    def call(self, input):
        tmp = self.inpu(input)
        for layer in self.dense_layers:
            tmp = layer(tmp)
        value = self.value(tmp)
        return value


class Agent:
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
        self.actor = self.init_a()
        self.critic = self.init_c()
    

    def init_a(self):
        actor_path = "./actor.tf"
        if exists(actor_path):
            return tf.keras.models.load_model(actor_path)
        else:
            return Actor(NUM_INPUTS, NUM_ACTIONS, ACTOR_HIDDEN)
    

    def init_c(self):
        critic_path = "./critic.tf"
        if exists(critic_path):
            return tf.keras.models.load_model(critic_path)
        else:
            return Critic(NUM_INPUTS, CRITIC_HIDDEN)


    def act(self, state: list[float]) -> int:
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        action_probabilities = self.actor(state).numpy()
        action = np.random.choice(NUM_ACTIONS, p=np.squeeze(action_probabilities))
        return action


    def actor_loss(self, action_probs, action, td) -> float:
        dist = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        a_loss = -log_prob*td
        return a_loss


    def learn(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        next_state = tf.convert_to_tensor(next_state)
        next_state = tf.expand_dims(next_state, 0)

        with tf.GradientTape() as t1, tf.GradientTape() as t2:
            action_probs = self.actor(state, training = True)
            value = self.critic(state, training = True)
            next_value = self.critic(next_state, training = True)
            td = reward + self.gamma * next_value * (1 - int(done)) - value
            actor_loss = self.actor_loss(action_probs, action, td)
            critic_loss = td**2
            
            actor_gradient = t1.gradient(actor_loss, self.actor.trainable_variables)
            critic_gradient = t2.gradient(critic_loss, self.critic.trainable_variables)
            self.actor_opt.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
            self.critic_opt.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))
            return actor_loss, critic_loss
