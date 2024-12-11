import numpy as np
import tensorflow as tf
import random
import traci
from collections import deque
import csv
import matplotlib.pyplot as plt
import math
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
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),#input as state size
            tf.keras.layers.Dense(24, activation='relu'),#hidden layer
            tf.keras.layers.Dense(self.action_size, activation='linear')#output layer
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def act(self, state):#epsilon greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size): #experience replay
        minibatch = random.sample(self.memory, batch_size) #randomly sample from memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))#Q-learning target
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target#update the target
            self.model.fit(state, target_f, epochs=1, verbose=0)#fit the model
            
        if self.epsilon > self.epsilon_min:
            self.epsilon_decay = math.exp(math.log(self.epsilon_min/self.epsilon)/ 100)#decay epsilon
            
            self.epsilon *= max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# functions to interact with SUMO
def get_traffic_state():
    lanes = traci.lane.getIDList()  # Get all lanes in the simulation
    state = []

    for lane in lanes:
        num_vehicles = traci.lane.getLastStepVehicleNumber(lane)
        waiting_time = traci.lane.getWaitingTime(lane)
        traffic_light_phase = traci.trafficlight.getPhase("J1")  # Junction_ID
        state.extend([num_vehicles, waiting_time, traffic_light_phase])

    return state

def apply_action(action):#apply action to the traffic light
    traffic_light_id = "J1"
    if action == 0:
        traci.trafficlight.setPhase(traffic_light_id, 0)
    elif action == 1:
        traci.trafficlight.setPhase(traffic_light_id, 1)
    elif action == 2:
        traci.trafficlight.setPhase(traffic_light_id, 2)
    elif action == 3:
        traci.trafficlight.setPhase(traffic_light_id, 3)

def compute_reward():
    total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())
    return -total_waiting_time  # Negative reward for waiting time at the intersection

def track_co2_emissions():
    lanes = traci.lane.getIDList()  # Get all lanes in the simulation
    total_co2_stopped = 0  # Initialize the total CO2 emissions for stopped vehicles
    
    for lane in lanes:
        # Get the IDs of all vehicles in the lane at this simulation step
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
        
        for vehicle_id in vehicle_ids:
            # Get the speed of the vehicle
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            
            # If the vehicle is stopped, calculate the CO2 emissions
            if vehicle_speed == 0:
                total_co2_stopped += traci.vehicle.getCO2Emission(vehicle_id)
    
    return total_co2_stopped

def save_results_to_csv(filename, data):
    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

def save_model(agent, filename="traffic_model.keras"):
    agent.model.save(filename)

# Run the RL simulation
def run_rl_simulation():
    sumo_binary = "sumo"  # "sumo-gui" for GUI, "sumo" for without GUI
    sumo_config = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 3/General Elective/Smart Mobility/My Paper/New/RL--based-Traffic-Intersection-Management/Main_Simulation.sumocfg"  # Replace with your SUMO config file path

    traci.start([sumo_binary, "-c", sumo_config, "--no-step-log", "true", "--log", "false"])

    avg_waiting_times_per_episode = []  # To store average waiting times per episode
    avg_co2_emissions_per_episode = []  # To store average CO2 emissions per episode

    state_size = 108  # Update based on features from get_traffic_state
    action_size = 4  # 4 actions (4 phases)
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    episodes = 100  # Total episodes to run

    for episode in range(episodes):
        print(f"Starting episode {episode + 1}/{episodes}")
        traci.load(["-c", sumo_config])
        step = 0
        state = np.array([get_traffic_state()])
        episode_waiting_times = []
        episode_co2_emissions = []

        while step < 100:
            step += 1
            traci.simulationStep()

            print(f"Episode {episode + 1}/{episodes}, Step {step}/100")
            action = agent.act(state)
            apply_action(action)
            reward = compute_reward()
            next_state = np.array([get_traffic_state()])
            done = False
            
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(agent.memory) > 32:
                agent.replay(32)

            # Track waiting time and CO2 emissions for the current step
            total_waiting_time = sum(get_traffic_state()[1::3])  # Summing waiting times from state
            total_co2_stopped = track_co2_emissions()
            episode_waiting_times.append(total_waiting_time)
            episode_co2_emissions.append(total_co2_stopped)

        avg_waiting_time_rl = np.mean(episode_waiting_times)
        avg_co2_emissions_rl = np.mean(episode_co2_emissions)
        avg_waiting_times_per_episode.append(avg_waiting_time_rl)
        avg_co2_emissions_per_episode.append(avg_co2_emissions_rl)

        print(f"Episode {episode + 1} completed. Average waiting time: {avg_waiting_time_rl}, Average CO2 emissions: {avg_co2_emissions_rl}")

        save_results_to_csv("waiting_times_rl.csv", [avg_waiting_time_rl])
        save_results_to_csv("co2_emissions_rl.csv", [avg_co2_emissions_rl])

    save_model(agent)
    traci.close()

    # Plot average waiting time
    plt.plot(range(1, episodes + 1), avg_waiting_times_per_episode, marker='o', label="Waiting Time")
    plt.title('Average Waiting Time per Episode_rl')
    plt.xlabel('Episode')
    plt.ylabel('Average Waiting Time')
    plt.grid()
    plt.legend()
    plt.savefig("average_waiting_time_rl.png", dpi=300)
    plt.show()

    # Plot average CO2 emissions
    plt.plot(range(1, episodes + 1), avg_co2_emissions_per_episode, marker='o', color='orange', label="CO2 Emissions")
    plt.title('Average CO2 Emissions per Episode_rl')
    plt.xlabel('Episode')
    plt.ylabel('Average CO2 Emissions')
    plt.grid()
    plt.legend()
    plt.savefig("average_co2_emissions_rl.png", dpi=300)
    plt.show()

# Run the RL simulation
if __name__ == "__main__":
    run_rl_simulation()
