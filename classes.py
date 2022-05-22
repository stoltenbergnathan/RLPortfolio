from enum import Enum
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataFunctions import *
from os.path import exists

NUM_INPUTS = 10
NUM_HIDDEN = 100
NUM_ACTIONS = 2
LEARNING_RATE = 0.0005


class Actions(Enum):
    INCREASE = 0
    DECREASE = 1


class State:
    def __init__(self, highs) -> None:
        self.highs = highs


class Env:
    def __init__(self, ticker: str) -> None:
        self.history = get_percent_history(ticker)
        self.max = NUM_INPUTS

    def do_something(self, action: int):
        if len(self.history) - self.max < NUM_INPUTS:
            return State([0.0] * NUM_INPUTS), 0, True
        else:
            new_state = State(self.history[self.max-NUM_INPUTS:self.max])
            self.max += 1
            new_change = new_state.highs[-1]
            if action == Actions.DECREASE.value and new_change < 0.0:
                reward = abs(new_change)
            elif action == Actions.INCREASE.value and new_change > 0.0:
                reward = new_change
            else:
                reward = -abs(new_change)
            return new_state, reward, False

            
    def get_init_state(self) -> State:
        state = State(self.history[0:self.max])
        self.max += 1
        return state


class Actor(tf.keras.Model):
    def __init__(self, num_inputs, num_actions, num_hidden) -> None:
        super().__init__()
        self.inpu = tf.keras.layers.Dense(num_inputs, activation='relu')
        self.h1 = tf.keras.layers.Dense(num_hidden, activation='relu')
        self.actions = tf.keras.layers.Dense(num_actions, activation = 'softmax')
    
    def call(self, input):
        tmp = self.inpu(input)
        tmp = self.h1(tmp)
        actions = self.actions(tmp)
        return actions


class Critic(tf.keras.Model):
    def __init__(self, num_inputs, num_hidden) -> None:
        super().__init__()
        self.inpu = tf.keras.layers.Dense(num_inputs, activation='relu')
        self.h1 = tf.keras.layers.Dense(num_hidden, activation='relu')
        self.value = tf.keras.layers.Dense(1, activation = None)
    
    def call(self, input):
        tmp = self.inpu(input)
        tmp = self.h1(tmp)
        value = self.value(tmp)
        return value


class Agent:
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
        # If model folders exist use them
        self.actor = self.init_a()
        self.critic = self.init_c()
    

    def init_a(self):
        actor_path = "./actor.tf"
        if exists(actor_path):
            return tf.keras.models.load_model(actor_path)
        else:
            return Actor(NUM_INPUTS, NUM_ACTIONS, NUM_HIDDEN)
    

    def init_c(self):
        critic_path = "./critic.tf"
        if exists(critic_path):
            return tf.keras.models.load_model(critic_path)
        else:
            return Critic(NUM_INPUTS, NUM_HIDDEN)


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