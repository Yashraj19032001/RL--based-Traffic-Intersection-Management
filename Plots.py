# compare_simulations.py

import numpy as np
import matplotlib.pyplot as plt

# Load the waiting times from both simulations
waiting_times_induction = np.loadtxt("induction_loop_waiting_times.csv", delimiter=",")
# waiting_times_rl = np.loadtxt("rl_waiting_times.csv", delimiter=",")

# Calculate the average waiting times for each system
avg_waiting_time_induction = np.mean(waiting_times_induction)
# avg_waiting_time_rl = np.mean(waiting_times_rl)

# Plotting the comparison
# plt.bar(['Induction Loop', 'RL'], [ avg_waiting_time_rl])
plt.bar(['Induction Loop'], [avg_waiting_time_induction])
# plt.bar(['Induction Loop', 'RL'], [avg_waiting_time_induction, avg_waiting_time_rl])
plt.ylabel('Average Waiting Time (s)')
plt.title('Traffic Signal Performance Comparison')

# Display the plot
plt.show()
