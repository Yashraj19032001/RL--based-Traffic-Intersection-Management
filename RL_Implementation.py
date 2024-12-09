import numpy as np
import tensorflow as tf
import random
import traci
from collections import deque
import csv
import matplotlib.pyplot as plt

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define functions to interact with SUMO
def get_traffic_state():
    lanes = traci.lane.getIDList()  # Get all lanes in the simulation
    state = []

    for lane in lanes:
        num_vehicles = traci.lane.getLastStepVehicleNumber(lane)  # Number of vehicles in the lane at the last step
        state.append(num_vehicles)

        # Example: Add the waiting time for vehicles in this lane
        waiting_time = traci.lane.getWaitingTime(lane)
        state.append(waiting_time)

        # Optionally: Add traffic light phase (if applicable)
        traffic_light_phase = traci.trafficlight.getPhase("J1")  # Replace with your actual traffic light ID
        state.append(traffic_light_phase)

    # Normalize or reduce the number of features if needed (this example uses all features for simplicity)
    return state

def apply_action(action):
    # Example: Set the traffic light phases based on the action
    traffic_light_id = "J1"  # Replace with your actual traffic light ID

    if action == 0:  # Phase 0, green for some lanes
        traci.trafficlight.setPhase(traffic_light_id, 0)  # Change to phase 0
    elif action == 1:  # Phase 1, green for other lanes
        traci.trafficlight.setPhase(traffic_light_id, 1)  # Change to phase 1

def compute_reward():
    # Example reward function (modify as per the traffic flow criteria)
    # Reward based on average waiting time across all lanes
    total_waiting_time = 0
    lanes = traci.lane.getIDList()
    for lane in lanes:
        total_waiting_time += traci.lane.getWaitingTime(lane)

    # Normalize or scale the reward as necessary
    reward = -total_waiting_time  # Negative reward for high waiting times
    return reward

# Function to save the waiting times to a CSV file
def save_waiting_times(waiting_times, filename="rl_waiting_times.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(waiting_times)  # Save the waiting times for the current step

# Function to save the trained model
def save_model(agent, filename="traffic_model.keras"):
    agent.model.save(filename)  # Save the model to a file

# Run the RL simulation
def run_rl_simulation():
    sumo_binary = "sumo"  # or "sumo" for headless mode
    sumo_config = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 3/General Elective/Smart Mobility/My Paper/New/RL--based-Traffic-Intersection-Management/Main_Simulation.sumocfg"  # Replace with your SUMO config file path

    traci.start([sumo_binary, "-c", sumo_config, "--no-step-log", "true", "--log", "false"])

    avg_waiting_times_per_episode = []  # To store average waiting times per episode

    # Initialize the RL agent with updated state size
    state_size = 108  # Update this based on the total number of features from get_traffic_state
    action_size = 2  # Example: 2 actions (green phases)
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    episodes = 100  # Total episodes to run

    for episode in range(episodes):
        print(f"Starting episode {episode + 1}/{episodes}")
        traci.load(["-c", sumo_config])
        step = 0
        state = np.array([get_traffic_state()])  # Use get_traffic_state function to get initial state
        done = False
        episode_waiting_times = []  # To store waiting times for the episode

        while step < 100:  # Max steps per episode (simulation time steps)
            step += 1
            traci.simulationStep()  # Advance the simulation by 1 step

            # Print the number of time steps completed
            print(f"Episode {episode + 1}/{episodes}, Step {step}/100 ")

            action = agent.act(state)  # Get action from the agent
            apply_action(action)  # Apply the action (change traffic light phase)
            reward = compute_reward()  # Calculate the reward based on current simulation state
            next_state = np.array([get_traffic_state()])  # Get the next state
            done = False  # Define how to check if the episode is done, based on your simulation

            # Store experience in memory
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            # Replay experience and learn
            if len(agent.memory) > 32:  # Batch size for learning
                agent.replay(32)

            # Collect waiting times for the current step
            total_waiting_time = sum(get_traffic_state()[1::3])  # Sum up waiting times from the state
            episode_waiting_times.append(total_waiting_time)

        # Compute and store the average waiting time for the episode
        avg_waiting_time = np.mean(episode_waiting_times)
        avg_waiting_times_per_episode.append(avg_waiting_time)

        print(f"Episode {episode + 1} completed. Average waiting time: {avg_waiting_time}")

    # Save the trained model after the simulation
    save_model(agent)  # Save the model to a file (e.g., 'traffic_model.h5')

    # Close the SUMO simulation
    traci.close()

    # Plot the average waiting time per episode
    plt.plot(range(1, episodes + 1), avg_waiting_times_per_episode, marker='o')
    plt.title('Average Waiting Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Waiting Time')
    plt.grid()
    
    # Save the plot to the same directory as the results
    plot_filename = "average_waiting_time_per_episode.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as {plot_filename}")


    plt.show()
    

# Run the RL simulation
if __name__ == "__main__":
    run_rl_simulation()
