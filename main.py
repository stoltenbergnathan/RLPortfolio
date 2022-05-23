from matplotlib import pyplot as plt
from classes import *


EPISODES = 5
GAMMA = 0.99
TRAINING = True

def train(agent):
    total_avg_rewards = []
    for episode in range(EPISODES):
        episode_reward = []
        done = False
        env = Env("TSLA")
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

    ep = [i for i in range(EPISODES)]
    plt.plot(ep, total_avg_rewards, 'b')
    plt.title("avg reward Vs episodes")
    plt.xlabel("episodes")
    plt.ylabel("average reward per 100 episodes")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    agent = Agent(GAMMA)
    if TRAINING:
        train(agent)
    else:
        pass # TODO test the model
