from datetime import datetime
from matplotlib import pyplot as plt
from classes import *
from dataFunctions import *


EPISODES = 50
GAMMA = 0.99
TRAINING = False
SAVE = True
TRAIN_START = "2020-01-01"
TRAIN_END = "2021-01-01"
TEST_START = "2021-01-01"
TEST_END =  "20202-01-01"


def train(agent: Agent):
    start = datetime.now()
    total_avg_rewards = []
    for episode in range(EPISODES):
        episode_reward = []
        done = False
        env = Env("AAPL")
        state = env.get_init_state()
        total_reward = 0
        actor_loss_history = []
        critic_loss_history = []
        while not done:
            action = agent.act(state.state())

            next_state, reward, done = env.do_something(action)

            actor_loss, critic_loss = agent.learn(state.state(), action, reward, next_state.state(), done)

            actor_loss_history.append(actor_loss)
            critic_loss_history.append(critic_loss)

            state = next_state
            total_reward += reward

            if done:
                print(f"Reward after episode {episode} is {total_reward}")
                episode_reward.append(total_reward)
                total_avg_rewards.append(np.mean(episode_reward))

    if SAVE:
        agent.actor.save("actor.tf")
        agent.critic.save("critic.tf")

    end = datetime.now()
    print(f"""
Training Summary:
Time Elapsed: {end-start}
Episodes: {EPISODES}
ACTOR_HIDDEN: {ACTOR_HIDDEN}
CRITIC_HIDDEN: {CRITIC_HIDDEN}
LEARNING_RATE: {LEARNING_RATE}
LOOKBACK: {LOOKBACK}
GAMMA: {GAMMA}
""")

    ep = [i for i in range(EPISODES)]
    plt.plot(ep, total_avg_rewards, 'b')
    plt.title("avg reward Vs episodes")
    plt.xlabel("episodes")
    plt.ylabel("average reward per 100 episodes")
    plt.grid(True)
    plt.show()


def test(agent: Agent):
    env = Env("AAPL")
    state = env.get_init_state()
    correct_guesses = 0
    total_guesses = 0
    done = False
    while not done:
        action = agent.act(state.state())
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
