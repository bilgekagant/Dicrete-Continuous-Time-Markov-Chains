import numpy as np
import matplotlib.pyplot as plt

# Number of states
n_states = 5

# Minimum probability for transitions to absorbing states
min_prob = 0.1

# Create a template for the transition matrix
template = np.zeros((n_states, n_states))

# States 3 and 4 are absorbing
template[3, 3] = 1
template[4, 4] = 1

# Define possible transition matrices
transition_matrices = []

# Iterate over possible values for the remaining probabilities
for i in np.linspace(0, 1 - 2*min_prob, 11):
    for j in np.linspace(0, 1 - i - 2*min_prob, 11):
        for k in np.linspace(0, 1 - i - j - 2*min_prob, 11):
            # Fill in the remaining probabilities for the transient states
            template[0, :] = [i, j, 1 - i - j - 2*min_prob, min_prob, min_prob]
            template[1, :] = [k, 1 - i - k - 2*min_prob, i, min_prob, min_prob]
            template[2, :] = [1 - j - k - 2*min_prob, j, k, min_prob, min_prob]
            # Add this matrix to the list
            transition_matrices.append(template.copy())

# Choose a transition matrix
transition_matrix = transition_matrices[0]

def theoretical_results(transition_matrix):
    Q = transition_matrix[:3, :3]
    R = transition_matrix[:3, 3:]
    I = np.identity(len(Q))
    IQ_inv = np.linalg.inv(I - Q)

    ones = np.ones((3, 1))
    mean_times = np.matmul(IQ_inv, ones)
    absorption_probs = np.matmul(IQ_inv, R)

    return mean_times.flatten(), absorption_probs

def run_simulation(transition_matrix, n_simulations):
    initial_states = [0, 1, 2]
    mean_times = []
    absorption_probs = []

    for initial_state in initial_states:
        time_counter = 0
        absorption_counter = [0, 0]

        for _ in range(n_simulations):
            state = initial_state
            steps = 0

            while state < 3:
                state = np.random.choice(n_states, p=transition_matrix[state])
                steps += 1

            time_counter += steps
            absorption_counter[state - 3] += 1

        mean_time = time_counter / n_simulations
        absorption_prob = [count / n_simulations for count in absorption_counter]

        mean_times.append(mean_time)
        absorption_probs.append(absorption_prob)

    # Print results
    print("Mean times until absorption:", mean_times)
    print("Absorption probabilities:", absorption_probs)

    return mean_times, absorption_probs

    

n_replications_list = [10, 100, 1000, 10000]
errors_mean_time = []
errors_absorption_probs = []

theoretical_mean_times, theoretical_absorption_probs = theoretical_results(transition_matrix)

for n_replications in n_replications_list:
    simulated_mean_times, simulated_absorption_probs = run_simulation(transition_matrix, n_replications)

    error_mean_time = np.abs(np.array(theoretical_mean_times) - np.array(simulated_mean_times))
    error_absorption_probs = np.abs(np.array(theoretical_absorption_probs) - np.array(simulated_absorption_probs))

    errors_mean_time.append(error_mean_time)
    errors_absorption_probs.append(error_absorption_probs)

# Calculate mean error
mean_errors_mean_time = np.mean(errors_mean_time, axis=1)
mean_errors_absorption_probs = np.mean(errors_absorption_probs, axis=(1, 2))

# Plotting the errors
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(n_replications_list, mean_errors_mean_time, marker='o')
plt.xscale('log')
plt.title('Mean Error in Estimated Mean Time Until Absorption')
plt.xlabel('Number of Replications')
plt.ylabel('Mean Error')

plt.subplot(1, 2, 2)
plt.plot(n_replications_list, mean_errors_absorption_probs, marker='o')
plt.xscale('log')
plt.title('Mean Error in Estimated Absorption Probabilities')
plt.xlabel('Number of Replications')
plt.ylabel('Mean Error')

plt.tight_layout()
plt.show()



# Theoretical Solutions: In the case of Discrete Time Markov Chains, there are established formulas to calculate the mean time until absorption and the absorption probabilities.

# Mean time until absorption can be calculated by (I - Q)^-1 * 1 where Q is the sub-matrix of the transition matrix containing only transient states, I is the identity matrix of the same size as Q, and 1 is a column vector of ones.
# Absorption probabilities can be calculated by (I - Q)^-1 * R where R is the sub-matrix of the transition matrix containing probabilities from transient states to absorbing states.
# Simulation Results: These are the results we obtained from the simulation in the previous step.

# To compare these, we can run the simulation multiple times with different number of replications (e.g., 10, 100, 1000, 10000, etc.), calculate the mean time until absorption and absorption probabilities each time, and then compare these with the theoretical results.

# To track the error, we can calculate the absolute difference between the theoretical and simulation results, and plot this error against the number of replications. This will give us a sense of how the error changes as we increase the number of replications.

# So, in summary:

# Write a function to calculate the theoretical results.
# Modify the simulation function to take the number of replications as a parameter.
# Run the simulation multiple times with different numbers of replications.
# Each time, compare the simulation results with the theoretical results and track the error.
# Plot the error against the number of replications.