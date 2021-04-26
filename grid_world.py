import matplotlib.pyplot as plt
import numpy as np
import mdp as mdptoolbox
import time
import json

from grid import GridClass


# Adapted from https://github.com/siddharth691/Path-Planning-using-Markov-Decision-Process
def get_reward(startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy, rm):
    cur_row = startRow
    cur_col = startCol

    opt_row = [startRow]
    opt_col = [startCol]
    max_points = num_states
    cur_point = 0

    # print('current state  row: {}, col: {}'.format(self.startRow, self.startCol))
    total_reward = 0
    while (1):

        try:

            cur_state = int(m[0][cur_row][cur_col])

        except IndexError:

            print('Didnt converge')
            break

        cur_opt_action = optimal_policy[cur_state]
        total_reward += rm[cur_state][cur_opt_action]

        if (cur_opt_action == 0):
            cur_row = cur_row
            cur_col = cur_col + 1
        elif (cur_opt_action == 1):
            cur_row = cur_row - 1
            cur_col = cur_col
        elif (cur_opt_action == 2):
            cur_row = cur_row
            cur_col = cur_col - 1
        else:
            cur_row = cur_row + 1
            cur_col = cur_col

        # Printing optimal action
        # print('Action: {}'.format(cur_opt_action))
        # print('Optimal Utility value for this state and this action : {}'.format(self.expected_values[int(cur_state)]))
        # print('Transition probability for current state and current action : {}'.format(self.st[cur_opt_action][int(cur_state)][int(self.m[0][cur_row][cur_col])]))
        # print('Reward for current state to perform current action : {}'.format(self.rm[int(cur_state)][cur_opt_action]))

        opt_row.append(cur_row)
        opt_col.append(cur_col)

        cur_point += 1

        if (cur_row == goalRow):
            if (cur_col == goalCol):
                print('Goal Reached!!')
                break

        if (cur_point == max_points):
            print('Steps limit over!!')
            break

    return total_reward


def visualize_path(startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy, rm, algorithm):
    # Visualize world
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.ion()
    ax.scatter(oCol, oRow, marker='s', s=500, c='gray')
    ax.scatter(startCol, startRow, s=500, c='b')
    ax.scatter(goalCol, goalRow, s=500, c='g')
    fig.savefig('grid_world.png')

    # Visualize path
    cur_row = startRow
    cur_col = startCol

    opt_row = [startRow]
    opt_col = [startCol]
    max_points = num_states
    cur_point = 0


    while 1:

        cur_state = int(m[0][cur_row][cur_col])
        cur_opt_action = optimal_policy[str(cur_state)]

        if cur_opt_action == 0:
            cur_row = cur_row
            cur_col = cur_col + 1
        elif cur_opt_action == 1:
            cur_row = cur_row - 1
            cur_col = cur_col
        elif cur_opt_action == 2:
            cur_row = cur_row
            cur_col = cur_col - 1
        else:
            cur_row = cur_row + 1
            cur_col = cur_col

        opt_row.append(cur_row)
        opt_col.append(cur_col)
        ax.plot(opt_col, opt_row, linewidth=5, color='red')
        plt.pause(0.1)

        cur_point += 1

        if cur_row == goalRow:
            if cur_col == goalCol:
                print('Goal Reached!!')
                break

        if cur_point == max_points:
            print('Steps limit over!!')
            break

    figname = algorithm
    fig.savefig(figname)
    plt.close()


def visualize_policy(maxRow, maxCol, startRow, startCol, goalRow, goalCol, oCol, oRow, num_states, m, optimal_policy,
                     rm, algorithm):
    # Visualize world
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    plt.ion()
    ax.scatter(oCol, oRow, marker='s', s=500, c='gray')
    ax.scatter(startCol, startRow, s=500, c='b')
    ax.scatter(goalCol, goalRow, s=500, c='g')

    # Arrow config
    arrow_head_len = 0.1
    len_arrow = 1 - 2 * arrow_head_len
    arrow = {'right': {'sx': -1 * (len_arrow / 2), 'sy': 0, 'dx': len_arrow, 'dy': 0},
             'top': {'sx': 0, 'sy': (len_arrow / 2), 'dx': 0, 'dy': -1 * len_arrow},
             'left': {'sx': (len_arrow / 2), 'sy': 0, 'dx': -1 * len_arrow, 'dy': 0},
             'bottom': {'sx': 0, 'sy': -1 * (len_arrow / 2), 'dx': 0, 'dy': len_arrow}}

    for cur_row in range(maxRow):
        for cur_col in range(maxCol):

            cur_state = int(m[0][cur_row][cur_col])
            cur_opt_action = optimal_policy[str(cur_state)]

            if cur_opt_action == 0:
                direction = 'right'
                x = arrow[direction]['sx'] + cur_col
                y = arrow[direction]['sy'] + cur_row
                dx = arrow[direction]['dx']
                dy = arrow[direction]['dy']

            elif cur_opt_action == 1:
                direction = 'top'
                x = arrow[direction]['sx'] + cur_col
                y = arrow[direction]['sy'] + cur_row
                dx = arrow[direction]['dx']
                dy = arrow[direction]['dy']

            elif cur_opt_action == 2:
                direction = 'left'
                x = arrow[direction]['sx'] + cur_col
                y = arrow[direction]['sy'] + cur_row
                dx = arrow[direction]['dx']
                dy = arrow[direction]['dy']

            else:
                direction = 'bottom'
                x = arrow[direction]['sx'] + cur_col
                y = arrow[direction]['sy'] + cur_row
                dx = arrow[direction]['dx']
                dy = arrow[direction]['dy']

            ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.1, fc='k', ec='k')

    figname = 'policy_' + algorithm
    fig.savefig(figname)
    plt.close()

# Adapted from https://github.com/siddharth691/Path-Planning-using-Markov-Decision-Process
def Q_learning(st, rm, m, goalRow, goalCol, gamma, num_states, iters):
    with open('grid_data_dump.txt') as json_data:
        data_world = json.load(json_data)

    time_total = []
    reward = []
    opt_policy = []
    mean_discrepancy = []
    for epsilon in [0.1, 0.3, 0.5, 0.7, 0.9]:

        print('Current Epsilon: {}'.format(epsilon))
        tot_time_start = time.time()
        ql = mdptoolbox.QLearning(st, rm, gamma, n_iter=iters)
        policies, rewardi = ql.run(epsilon, 0.3, m[0][goalRow][goalCol], m, decay_eps=0, decay_alpha=0)
        tot_time_end = time.time()
        tot_time = tot_time_end - tot_time_start
        time_total.append(tot_time)

        mean_discrepancy.append(rewardi)
        optimal_policy = ql.policy
        optimal_policy = tuple(int(x) for x in optimal_policy)
        optimal_policy = dict(zip([str(x) for x in list(range(num_states))], list(optimal_policy)))
        policies = [tuple(int(x) for x in opt_policy) for opt_policy in policies]
        policies = [dict(zip(list(range(num_states)), list(opt_policy))) for opt_policy in policies]

        reward_policy = []
        for p_pol in policies:
            reward_p = get_reward(data_world['startRow'], data_world['startCol'], data_world['goalRow'],
                                  data_world['goalCol'], data_world['oCol'], data_world['oRow'],
                                  data_world['num_states'], data_world['m'], p_pol, data_world['rm'])
            reward_policy.append(reward_p)

        reward.append(reward_policy)
        opt_policy.append(optimal_policy)

    return time_total, opt_policy, reward, mean_discrepancy


# Adapted from https://github.com/siddharth691/Path-Planning-using-Markov-Decision-Process
def policy_iteration(st, rm, gamma, num_states):
    iterations = list(range(1, 1000, 10))
    data_policy = {'convergence': {}}

    for iter in iterations:

        print('Current Iteration: {}'.format(iter))

        data_policy[str(iter)] = {}

        tot_time_start = time.time()
        vi = mdptoolbox.PolicyIteration(st, rm, gamma, max_iter=1000, eval_type=1)
        time_iter, iter_value, iter_policy, policy_change, policies = vi.run(max_iter=iter)
        tot_time_end = time.time()
        tot_time = tot_time_end - tot_time_start

        policy_change = [int(x) for x in policy_change]
        if np.any(np.array(iter_value) > iter):
            raise ValueError('Value loop of Policy Iteration not stopping at maximum iterations provided')

        data_policy[str(iter)]['tot_time'] = tot_time
        data_policy[str(iter)]['time_iter'] = time_iter
        data_policy[str(iter)]['policy_iter'] = iter_policy
        data_policy[str(iter)]['value_iter'] = iter_value
        data_policy[str(iter)]['policy_change'] = policy_change

    print('Convergence')
    tot_time_start = time.time()
    vi = mdptoolbox.PolicyIteration(st, rm, gamma, max_iter=10000, eval_type=1)
    time_iter, iter_value, iter_policy_policy, policy_change, policies = vi.run(max_iter=10000)
    tot_time_end = time.time()

    policy_change = [int(x) for x in policy_change]
    policies = [tuple(int(x) for x in opt_policy) for opt_policy in policies]
    optimal_policy = vi.policy
    expected_values = vi.V
    optimal_policy = tuple(int(x) for x in optimal_policy)
    expected_values = tuple(float(x) for x in expected_values)

    optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
    expected_values = list(expected_values)
    policies = [dict(zip(list(range(num_states)), list(opt_policy))) for opt_policy in policies]

    data_policy['convergence']['tot_time'] = tot_time_end - tot_time_start
    data_policy['convergence']['time_iter'] = time_iter
    data_policy['convergence']['policy_iter'] = iter_policy_policy
    data_policy['convergence']['value_iter'] = iter_value
    data_policy['convergence']['policy_change'] = policy_change
    data_policy['convergence']['optimal_policy'] = optimal_policy
    data_policy['convergence']['expected_values'] = expected_values
    data_policy['convergence']['policies'] = policies

    return data_policy


def dump_data(data, file_name):
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)


# Adapted from https://github.com/siddharth691/Path-Planning-using-Markov-Decision-Process
def value_iteration(st, rm, gamma, num_states):
    iterations = list(range(1, 1000, 10))
    data_value = {'convergence': {}}
    for iter in iterations:

        print('Current Iteration: {}'.format(iter))
        data_value[str(iter)] = {}

        tot_time_start = time.time()
        vi = mdptoolbox.ValueIteration(st, rm, gamma, max_iter=1000, epsilon=0.0001)
        # vi.setVerbose()
        time_iter, iter_value, variation, policies = vi.run(max_iter=iter)
        tot_time_end = time.time()
        tot_time = tot_time_end - tot_time_start

        if (iter_value > iter):
            raise ValueError('ValueIteration is not stopping at maximum iterations')

        data_value[str(iter)]['tot_time'] = tot_time
        data_value[str(iter)]['time_iter'] = time_iter
        data_value[str(iter)]['value_iter'] = iter_value
        data_value[str(iter)]['variation'] = variation

    print('Convergence')
    tot_time_start = time.time()
    vi = mdptoolbox.ValueIteration(st, rm, gamma, max_iter=10000, epsilon=0.0001)
    time_iter, iter_value, variation, policies = vi.run(max_iter=10000)
    tot_time_end = time.time()

    optimal_policy = vi.policy
    expected_values = vi.V
    policies = [tuple(int(x) for x in opt_policy) for opt_policy in policies]
    optimal_policy = tuple(int(x) for x in optimal_policy)
    expected_values = tuple(float(x) for x in expected_values)

    optimal_policy = dict(zip(list(range(num_states)), list(optimal_policy)))
    expected_values = list(expected_values)
    policies = [dict(zip(list(range(num_states)), list(opt_policy))) for opt_policy in policies]

    data_value['convergence']['tot_time'] = tot_time_end - tot_time_start
    data_value['convergence']['time_iter'] = time_iter
    data_value['convergence']['value_iter'] = iter_value
    data_value['convergence']['variation'] = variation
    data_value['convergence']['optimal_policy'] = optimal_policy
    data_value['convergence']['expected_values'] = expected_values
    data_value['convergence']['policies'] = policies

    return data_value


def plot_grid_data(file_data_world, file_data_value, file_data_policy):
    iterations = list(range(1, 1000, 10))
    with open(file_data_world) as json_data:
        data_world = json.load(json_data)

    with open(file_data_value) as json_data:
        data_value = json.load(json_data)

    with open(file_data_policy) as json_data:
        data_policy = json.load(json_data)

    # Total computation time
    tot_time_policy = []
    tot_time_value = []
    for iter in iterations:
        tot_time_policy.append(data_policy[str(iter)]['tot_time'])
        tot_time_value.append(data_value[str(iter)]['tot_time'])

    fig, ax = plt.subplots()
    ax.plot(iterations, tot_time_policy)
    ax.plot(iterations, tot_time_value)
    plt.legend(['PolicyIteration', 'ValueIteration'])
    plt.title('Total computation time (s)')
    plt.ylabel('Time in seconds')
    plt.xlabel('Iterations')
    fig.savefig('total_time.png')

    # Average value update time
    value_time_policy = []
    value_time_value = []

    for itr in iterations:

        time_policy = []
        for time_value in data_policy[str(itr)]['time_iter']:
            time_policy.append(sum(time_value) / float(len(time_value)))

        value_time_policy.append(sum(time_policy) / len(time_policy))
        value_time_value.append(sum(data_value[str(itr)]['time_iter']) / len(data_value[str(itr)]['time_iter']))

    fig2, ax2 = plt.subplots()
    ax2.plot(iterations, value_time_policy)
    ax2.plot(iterations, value_time_value)
    ax2.legend(['PolicyIteration', 'ValueIteration'])
    plt.title('Average time of per value update iteration')
    plt.ylabel('Time in seconds')
    plt.xlabel('Iterations')
    fig2.savefig('value_time.png')

    # Visualize path and policy for policy iteration and value iteration
    visualize_path(data_world['startRow'], data_world['startCol'], data_world['goalRow'], data_world['goalCol'],
                   data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'],
                   data_policy['convergence']['optimal_policy'], data_world['rm'], 'policy_iteration')
    visualize_path(data_world['startRow'], data_world['startCol'], data_world['goalRow'], data_world['goalCol'],
                   data_world['oCol'], data_world['oRow'], data_world['num_states'], data_world['m'],
                   data_value['convergence']['optimal_policy'], data_world['rm'], 'value_iteration')
    visualize_policy(data_world['maxRow'], data_world['maxCol'], data_world['startRow'], data_world['startCol'],
                     data_world['goalRow'], data_world['goalCol'], data_world['oCol'], data_world['oRow'],
                     data_world['num_states'], data_world['m'], data_policy['convergence']['optimal_policy'],
                     data_world['rm'], 'policy_iteration')
    visualize_policy(data_world['maxRow'], data_world['maxCol'], data_world['startRow'], data_world['startCol'],
                     data_world['goalRow'], data_world['goalCol'], data_world['oCol'], data_world['oRow'],
                     data_world['num_states'], data_world['m'], data_value['convergence']['optimal_policy'],
                     data_world['rm'], 'value_iteration')

    # Calculating reward
    policy_policy = data_policy['convergence']['policies']
    policy_value = data_value['convergence']['policies']

    print(len(policy_policy))
    print(len(policy_value))
    reward_policy = []
    reward_value = []
    for p_pol in policy_policy:
        reward_p = get_reward(data_world['startRow'], data_world['startCol'], data_world['goalRow'],
                              data_world['goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'],
                              data_world['m'], p_pol, data_world['rm'])
        reward_policy.append(reward_p)

    for v_pol in policy_value:
        reward_v = get_reward(data_world['startRow'], data_world['startCol'], data_world['goalRow'],
                              data_world['goalCol'], data_world['oCol'], data_world['oRow'], data_world['num_states'],
                              data_world['m'], v_pol, data_world['rm'])
        reward_value.append(reward_v)

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.plot(list(range(len(reward_policy))), reward_policy)
    plt.xlabel('iterations')
    plt.ylabel('reward collected')
    plt.title('Policy Iteration')
    fig3.savefig('reward_policy.png')

    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.plot(list(range(len(reward_value)))[1:100], reward_value[1:100])
    plt.xlabel('iterations')
    plt.ylabel('reward collected')
    plt.title('Value Iteration')
    fig4.savefig('reward_value.png')


def main():
    grid = GridClass()
    grid.get_obstacles()
    grid.build_map()
    grid.build_st_trans_matrix()
    grid.build_reward_matrix()

    grid_data = {'st': grid.st.tolist(), 'rm': grid.rm.tolist(), 'gamma': grid.gamma, 'num_states': grid.num_states,
                 'startRow': grid.startRow, 'startCol': grid.startCol, 'goalRow': grid.goalRow, 'goalCol': grid.goalCol,
                 'oCol': grid.oCol, 'oRow': grid.oRow, 'm': grid.m.tolist(), 'maxRow': grid.maxRow,
                 'maxCol': grid.maxCol}

    with open('grid_data_dump.txt', 'w') as outfile:
        json.dump(grid_data, outfile)

    with open('grid_data_dump.txt') as json_data:
        grid_data = json.load(json_data)

    grid_data['st'] = np.array(grid_data['st'])
    grid_data['rm'] = np.array(grid_data['rm'])
    grid_data['m'] = np.array(grid_data['m'])

    ##Policy Iteration Experimentations
    policy_data = policy_iteration(grid_data['st'], grid_data['rm'], grid_data['gamma'], grid_data['num_states'])
    dump_data(policy_data, 'data_policy.txt')

    ##Value Iteration Experimentations
    data_value = value_iteration(grid_data['st'], grid_data['rm'], grid_data['gamma'], grid_data['num_states'])
    dump_data(data_value, 'data_value.txt')

    # Showing plots
    plot_grid_data('grid_data_dump.txt', 'data_value.txt', 'data_policy.txt')
    plt.show()

    with open('grid_data_dump.txt') as json_data:
        world_data = json.load(json_data)

    world_data['st'] = np.array(world_data['st'])
    world_data['rm'] = np.array(world_data['rm'])
    world_data['m'] = np.array(world_data['m'])
    iterations = 3000
    time_total, opt_policy, reward, mean_discrepancy = Q_learning(world_data['st'], world_data['rm'], world_data['m'],
                                                                  world_data['goalRow'], world_data['goalCol'],
                                                                  world_data['gamma'], world_data['num_states'], iterations)

    data_q = {'time_total': time_total, 'opt_policy': opt_policy, 'reward': reward,
              'mean_discrepancy': mean_discrepancy}

    with open('data_q.txt', 'w') as outfile:
        json.dump(data_q, outfile)

    with open('data_q.txt') as json_data:
        data_q = json.load(json_data)

    time_total = data_q['time_total']
    opt_policy = data_q['opt_policy']
    reward = data_q['reward']
    mean_discrepancy = data_q['mean_discrepancy']

    # Plot total time with epsilon
    fig1, ax1 = plt.subplots()
    plt.plot([0.1, 0.3, 0.5, 0.7, 0.9], time_total)
    plt.xlabel('Epsilon')
    plt.ylabel('Total Computation time (s)')
    plt.title('Q-Learning total time for %d' % iterations)
    fig1.savefig('ComputationTime-Q.png')

    # Plot reward with iterations for each epsilon

    fig2, ax2 = plt.subplots()
    for index, eps in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        ax2.plot(list(range(1, iterations + 1)), reward[index])

    plt.xlabel('iterations')
    plt.ylabel('Reward collected')
    plt.title('Reward collected varying with iterations (epsilon = 0.9)')
    plt.legend(['eps = 0.1', 'eps = 0.3', 'eps = 0.5', 'eps = 0.7', 'eps = 0.9'])
    plt.ylim((-100, 100))
    fig2.savefig('reward_iter_eps-Q.png')

    # Mean discrepancy with iterations for each epsilon

    fig3, ax3 = plt.subplots()
    for index, eps in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        ax3.plot(list(range(1, iterations + 1)), mean_discrepancy[index])

    plt.xlabel('iterations')
    plt.ylabel('Mean of discrepancy (= alpha*delta)')
    plt.title('Mean discrepancy')
    plt.legend(['eps = 0.1', 'eps = 0.3', 'eps = 0.5', 'eps = 0.7', 'eps = 0.9'])
    fig3.savefig('mean_discrepancy-Q.png')

    for index, eps in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        policy = opt_policy[index]
        algorithm = 'q-learning' + str(eps)
        visualize_policy(world_data['maxRow'], world_data['maxCol'], world_data['startRow'], world_data['startCol'],
                         world_data['goalRow'], world_data['goalCol'], world_data['oCol'], world_data['oRow'],
                         world_data['num_states'], world_data['m'], policy, world_data['rm'], algorithm, eps)
        visualize_path(world_data['startRow'], world_data['startCol'], world_data['goalRow'], world_data['goalCol'],
                       world_data['oCol'], world_data['oRow'], world_data['num_states'], world_data['m'], policy,
                       world_data['rm'], algorithm, eps)

    plt.show()


if __name__ == '__main__':
    main()
