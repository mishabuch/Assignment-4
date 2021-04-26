import time
import matplotlib.pyplot as plt
import numpy as np
import mdptoolbox.example
from forest_plots import plot_forest_management


def Forest_Experiments():
    print('Forest management - Policy iteration')
    P, R = mdptoolbox.example.forest(S=1000)
    rewards, policy, iterations, time_taken, gamma_values = [0] * 20, [0] * 20, [0] * 20, [0] * 20, [0] * 20
    for i in range(0, 20):
        gamma_values[i] = discount = (i + 0.5) / 20
        pi = mdptoolbox.mdp.PolicyIteration(P, R, discount)
        pi.run()
        rewards[i], policy[i], iterations[i], time_taken[i] = np.mean(pi.V), pi.policy, pi.iter, pi.time

    plot_forest_management(gamma_values, iterations, time_taken, rewards, 'Forest Management - Policy Iteration')

    print('Forest management - Value iteration')
    for i in range(0, 20):
        gamma_values[i] = discount = (i + 0.5) / 20
        pi = mdptoolbox.mdp.ValueIteration(P, R, discount)
        pi.run()
        rewards[i], policy[i], iterations[i], time_taken[i] = np.mean(pi.V), pi.policy, pi.iter, pi.time

    plot_forest_management(gamma_values, iterations, time_taken, rewards, 'Forest Management - Value Iteration')

    print('Forest management - Q Learning')
    policy = []
    time_taken = []
    Q_table = []
    rewards = []
    epsilon_values = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.95]
    for epsilon in epsilon_values:
        st = time.time()
        pi = mdptoolbox.mdp.QLearning(P, R, 0.95)
        pi.run(epsilon)
        end = time.time()
        policy.append(pi.policy)
        rewards.append(pi.reward_array)
        rewards.append(np.mean(pi.V))
        time_taken.append(end - st)
        Q_table.append(pi.Q)

    for i in range(0, len(rewards)):
        plt.plot(range(0, 10000), rewards[i], label='epsilon={}'.format(epsilon_values[i]))
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title('Forest Management - Q Learning - Epsilon vs Rewards')
    plt.ylabel('Average Reward')
    plt.show()
    return


Forest_Experiments()
