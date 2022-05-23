from datetime import datetime
import random
from matplotlib import pyplot as plt
from classes import *
from dataFunctions import *


EPISODES = 100
GAMMA = 0.99
TRAINING = True
TICKERS = get_SP_tickers()


def get_ticker() -> list[float]:
    good = False
    while not good:
        ticker = random.choice(TICKERS)
        p_hist = get_percent_history(ticker)
        if p_hist:
            good = True
    return p_hist


def train(agent):
    start = datetime.now()
    total_avg_rewards = []
    for episode in range(EPISODES):
        episode_reward = []
        done = False
        env = Env(get_percent_history("TSLA"))
        state = env.get_init_state()
        total_reward = 0
        actor_loss_history = []
        critic_loss_history = []
        while not done:
            action = agent.act(state.highs)

            next_state, reward, done = env.do_something(action)

            actor_loss, critic_loss = agent.learn(state.highs, action, reward, next_state.highs, done)

            actor_loss_history.append(actor_loss)
            critic_loss_history.append(critic_loss)

            state = next_state
            total_reward += reward

            if done:
                print(f"Reward after episode {episode} is {total_reward}")
                episode_reward.append(total_reward)
                total_avg_rewards.append(np.mean(episode_reward))

    agent.actor.save("actor.tf")
    agent.critic.save("critic.tf")

    end = datetime.now()
    print(f"""
    Training Summary:\n
    Time Elapsed: {end-start}\n
    Episodes: {EPISODES}\n
    ACTOR_HIDDEN: {ACTOR_HIDDEN}\n
    CRITIC_HIDDEN: {CRITIC_HIDDEN}\n
    LEARNING_RATE: {LEARNING_RATE}\n
    GAMMA: {GAMMA}
    """)

    ep = [i for i in range(EPISODES)]
    plt.plot(ep, total_avg_rewards, 'b')
    plt.title("avg reward Vs episodes")
    plt.xlabel("episodes")
    plt.ylabel("average reward per 100 episodes")
    plt.grid(True)
    plt.show()


def test(agent):
    env = Env(get_percent_history("TSLA"))
    state = env.get_init_state()
    correct_guesses = 0
    total_guesses = 0
    done = False
    while not done:
        action = agent.act(state.highs)
        expected_action, state, done = env.expected_action()
        if not done:
            if expected_action == action:
                correct_guesses += 1
            total_guesses += 1
    
    print(f"The agent correctly predicted the INCREASE / DECREASE {correct_guesses/total_guesses} percent of the time")



if __name__ == "__main__":
    agent = Agent(GAMMA)
    if TRAINING:
        train(agent)
    else:
        test(agent)
