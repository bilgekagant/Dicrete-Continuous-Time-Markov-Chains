# It's a simulation of a continuous time Markov chain (CTMC) with 5 states, 
# and the printouts are showing the time at which the process transitions to a new state and the new state itself.
import numpy as np
import matplotlib.pyplot as plt

# Number of states
n_states = 5

# Rate of transition from each state to every other state
rate = 1

# Construct the rate matrix
rate_matrix = np.full((n_states, n_states), rate)

# Each diagonal entry is the negative of the sum of the off-diagonal entries in the same row
np.fill_diagonal(rate_matrix, -rate*(n_states-1))

def run_simulation(rate_matrix, initial_state, max_time):
    # Initialize the state and time
    state = initial_state
    time = 0
    
    # Initialize a list to record the time spent in each state
    time_spent_in_each_state = np.zeros(n_states)
    previous_time = 0
    
    # Run the simulation until the maximum time
    while True:
        # Calculate the total rate of leaving the current state
        total_rate = -rate_matrix[state, state]

        # Generate the time until the next event
        time_until_next_event = np.random.exponential(1/total_rate)

        # Check if the next event exceeds max_time
        if time + time_until_next_event > max_time:
            # Add the remaining time to the current state
            time_spent_in_each_state[state] += max_time - time
            break
        else:
            # Update the time
            time += time_until_next_event

        # Add the time spent in the current state
        time_spent_in_each_state[state] += time - previous_time
        previous_time = time

        # Generate the type of the event (the state to transition to)
        rates = rate_matrix[state, :].copy()  # Create a copy of the rates array
        rates[state] = 0
        probs = rates / total_rate
        mask = rates != 0
        non_zero_states = np.arange(n_states)[mask]
        non_zero_probs = probs[mask]
        next_state = np.random.choice(non_zero_states, p=non_zero_probs)
        
        # Update the state
        state = next_state
    
    # Calculate the fraction of time spent in each state
    steady_state_distribution = time_spent_in_each_state / max_time
    
    return steady_state_distribution

# Run the simulation
initial_state = 0
max_time = 10
steady_state_distribution = run_simulation(rate_matrix, initial_state, max_time)

# The theoretical steady-state distribution for this CTMC 
# should be a uniform distribution, because all states are equally 
# likely in the long run given that all states are in the same class 
# and the rate of transition from each state to every other state is the same.

#estimated steady-state diribution
print("Estimated steady-state distribution:", steady_state_distribution)

#estimated steady-state diribution compare to the theoretical steady-state distribution
theoretical_distribution = np.full(n_states, 1/n_states)
print("Theoretical steady-state distribution:", theoretical_distribution)

# the estimated steady-state distribution will get closer to the theoretical one 
# as max_time increases. The law of large numbers ensures this.

# Initialize list to store errors
errors = []

# Initialize list to store max_times
max_times = list(range(100, 1000, 100))

# Theoretical steady-state distribution
theoretical_distribution = np.full(n_states, 1/n_states)

for max_time in max_times:
    # Run the simulation
    steady_state_distribution = run_simulation(rate_matrix, initial_state, max_time)
    
    # Calculate the error
    error = np.abs(theoretical_distribution - steady_state_distribution)
    
    # Store the average error
    errors.append(np.mean(error))

# Plot the error over time
plt.plot(max_times, errors)
plt.xlabel('Simulation Time')
plt.ylabel('Average Absolute Error')
plt.title('Error in Estimating Steady-State Distribution Over Time')
plt.show()

##OUTLINE
# Here's an outline of how you can proceed:

# Compute Theoretical Solutions: For the given CTMC with all states in the same class 
# and equal rates of transition, the theoretical steady-state distribution is uniform.
#  This is because every state is equally likely in the long run. So, the probability of being 
# in any state is 1/n_states. We've already computed this in the previous discussions.

# Compute Simulation Estimates: You have already computed the simulation estimates in Step 2.

# Compare Theoretical Solutions and Simulation Estimates: You can calculate the error as 
# the absolute difference between the theoretical solutions and simulation estimates. Since 
# both are vectors (of size n_states), you need to compute the error for each state.

# Sketch the Error over Time: You can run multiple simulations with increasing 'max_time', 
# compute the error each time, and then plot the errors. This will show you how the error changes as the simulation run time increases.