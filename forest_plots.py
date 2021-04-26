from matplotlib import pyplot as plt


def plot_forest_management(gamma_values, iterations, time_array, rewards, title):
    plt.plot(gamma_values, rewards)
    plt.xlabel('Gamma Value')
    plt.ylabel('Rewards')
    plt.title('{} - Reward vs Gamma Values'.format(title))
    plt.grid()
    plt.show()

    plt.plot(gamma_values, iterations)
    plt.xlabel('Gamma Value')
    plt.ylabel('Iterations to Converge')
    plt.title('{} - Convergence vs Gamma Values'.format(title))
    plt.grid()
    plt.show()

    plt.plot(gamma_values, time_array)
    plt.xlabel('Gamma Value')
    plt.title('{} - Execution Time vs Gamma Values'.format(title))
    plt.ylabel('Time Taken (in seconds)')
    plt.grid()
    plt.show()