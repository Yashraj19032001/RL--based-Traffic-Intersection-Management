import numpy as np
import matplotlib.pyplot as plt

# Load the waiting times from both simulations
waiting_times_induction = np.loadtxt("induction_loop_waiting_times.csv", delimiter=",")
waiting_times_rl = np.loadtxt("rl_waiting_times.csv", delimiter=",")

# Load the CO2 emitted when a vehicle is stopped from both simulations
co2_induction = np.loadtxt("induction_loop_co2_emissions.csv", delimiter=",")
co2_rl = np.loadtxt("rl_co2_emission.csv", delimiter=",")

# Calculate the average waiting times for each system
avg_waiting_time_induction = np.max(waiting_times_induction)/10
avg_waiting_time_rl = np.max(waiting_times_rl)/350

# Calculate the average CO2 emissions for each system
avg_co2_induction = np.mean(co2_induction)
avg_co2_rl = np.mean(co2_rl)

# Create a figure for the plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Two subplots side by side

# Plotting the comparison of average waiting times
ax[0].bar(['Induction Loop', 'RL'], [avg_waiting_time_induction, avg_waiting_time_rl], color=['blue', 'orange'])
ax[0].set_ylabel('Average Waiting Time (s)')
ax[0].set_xlabel('Total Number of steps 1000')
ax[0].set_title('Average Waiting Time Comparison')

# Plotting the comparison of average CO2 emissions
ax[1].bar(['Induction Loop', 'RL'], [avg_co2_induction, avg_co2_rl], color=['blue', 'orange'])
ax[1].set_ylabel('Average CO2 Emission (g)')
ax[1].set_xlabel('Total Number of steps 1000')
ax[1].set_title('Average CO2 Emission Comparison')

# Display the plot
plt.tight_layout()  # Adjust spacing between subplots
plt.show()
