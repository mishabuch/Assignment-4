from matplotlib import pyplot as plt


def plot_forest_management(gamma_values, iterations, time_array, rewards, title):
    plt.plot(gamma_values, rewards)
    plt.ylabel('Rewards')
    plt.xlabel('Discount')
    plt.title('{} - Reward vs Discount'.format(title))
    plt.grid()
    plt.show()

    plt.plot(gamma_values, iterations)
    plt.ylabel('Iterations to Converge')
    plt.xlabel('Discount')
    plt.title('{} - Convergence vs Discount'.format(title))
    plt.grid()
    plt.show()

    plt.plot(gamma_values, time_array)
    plt.title('{} - Execution Time vs Discount'.format(title))
    plt.xlabel('Discount')
    plt.ylabel('Time Taken (in seconds)')
    plt.grid()
    plt.show()