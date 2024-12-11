import numpy as np
import tensorflow as tf
import traci
import csv
import matplotlib.pyplot as plt

# Load the saved model
def load_model(model_path="traffic_model.keras"):
    return tf.keras.models.load_model(model_path)

# Define functions to interact with SUMO
def get_traffic_state():
    lanes = traci.lane.getIDList()  # Get all lanes in the simulation
    state = []

    for lane in lanes:
        num_vehicles = traci.lane.getLastStepVehicleNumber(lane)  # Number of vehicles in the lane at the last step
        state.append(num_vehicles)

        waiting_time = traci.lane.getWaitingTime(lane)
        state.append(waiting_time)

        traffic_light_phase = traci.trafficlight.getPhase("J1")  # Junction_ID
        state.append(traffic_light_phase)

    return state

def apply_action(action):
    traffic_light_id = "J1"  # Junction_ID

    if action == 0:  # Phase 0, green for some lanes
        traci.trafficlight.setPhase(traffic_light_id, 0)  # Change to phase 0
    elif action == 1:  # Phase 1, green for other lanes
        traci.trafficlight.setPhase(traffic_light_id, 1)  # Change to phase 1
    elif action == 2:  # Phase 2, green for other lanes
        traci.trafficlight.setPhase(traffic_light_id, 2) # Change to phase 2
    elif action == 3:  # Phase 3, green for other lanes
        traci.trafficlight.setPhase(traffic_light_id, 3) # Change to phase 3
# Function to save the waiting times to a CSV file
def save_waiting_times(waiting_times, filename="rl_waiting_times.csv"):
    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(waiting_times)
    except Exception as e:
        print(f"Error saving waiting times: {e}")

# Function to save CO2 emissions to a CSV file
def save_co2_emissions(co2_emissions, filename="rl_co2_emission.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(co2_emissions)  # Save the CO2 emissions for the current step

# Function to calculate CO2 emissions
def get_total_co2_emission_at_traffic_light():
    vehicle_ids = traci.vehicle.getIDList()
    total_co2 = 0  # Initialize total CO2 emissions

    for veh in vehicle_ids:
        # Check if the vehicle is stopped (speed is close to 0)
        speed = traci.vehicle.getSpeed(veh)
        if speed < 0.01:  # Threshold for considering vehicle as stopped
            # Add CO2 emission of the vehicle if it is stopped
            total_co2 += traci.vehicle.getCO2Emission(veh)

    return total_co2

# Run the RL simulation to test the model
def run_test_simulation(model_path="traffic_model.keras"):
    sumo_binary = "sumo"  # or "sumo" for headless mode
    sumo_config = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 3/General Elective/Smart Mobility/My Paper/New/RL--based-Traffic-Intersection-Management/Main_Simulation.sumocfg"  # Replace with your SUMO config file path

    traci.start([sumo_binary, "-c", sumo_config, "--no-step-log", "true", "--log", "false"])

    # Load the saved model
    model = load_model(model_path)

    steps = 1001  # Number of simulation steps for testing
    batch_size = 100  # Batch size for averaging waiting time and emissions
    waiting_times_per_batch = []  # Store average waiting times for each batch
    co2_emissions = []  # Store CO2 emissions over time
    current_batch_waiting_times = []  # Store waiting times within the current batch
    current_batch_co2_emissions = []  # Store CO2 emissions within the current batch

    for step in range(steps):
        traci.simulationStep()  # Advance the simulation by 1 step

        # Get the current state from the simulation
        state = np.array([get_traffic_state()])

        # Predict the best action using the loaded model
        action = np.argmax(model.predict(state))  # Predict action based on the state

        # Apply the predicted action (change the traffic light phase)
        apply_action(action)

        # Collect the total waiting time for all vehicles in the simulation
        total_waiting_time = sum(get_traffic_state()[1::3])  # Extract waiting times from the state
        current_batch_waiting_times.append(total_waiting_time)

        # Record CO2 emissions
        total_co2 = get_total_co2_emission_at_traffic_light()
        co2_emissions.append(total_co2)
        current_batch_co2_emissions.append(total_co2)

        # If batch is complete, calculate the average waiting time and CO2 emissions for the batch
        if (step + 1) % batch_size == 0:
            avg_waiting_time = np.mean(current_batch_waiting_times)
            waiting_times_per_batch.append(avg_waiting_time)
            
            save_waiting_times([avg_waiting_time])  # Save average waiting times
            # Save the average CO2 emissions for this batch
            avg_co2_emissions = np.mean(current_batch_co2_emissions)
            save_co2_emissions([avg_co2_emissions])  # Save the CO2 emissions of this batch

            # Reset for the next batch
            current_batch_waiting_times = []  
            current_batch_co2_emissions = []  

        # Print the progress
        print(f"Step {step + 1}/{steps} - Action: {action} - CO2 Emission: {total_co2}")

    traci.close()

    # Plot the average waiting time per batch
    plt.figure()
    plt.plot(range(1, len(waiting_times_per_batch) + 1), waiting_times_per_batch, marker='o')
    plt.title('Average Waiting Time per 100 Steps')
    plt.xlabel('Batch (100 Steps per Batch)')
    plt.ylabel('Average Waiting Time')
    plt.grid()
    plt.savefig("average_waiting_time_per_batch.png")

    # Plot CO2 emissions over time
    plt.figure()
    plt.plot(range(1, steps + 1), co2_emissions, marker='o', markersize=2, alpha=0.7)
    plt.title('CO2 Emissions Over Time (RL)')
    plt.xlabel('Simulation Steps')
    plt.ylabel('CO2 Emission (g)')
    plt.grid()
    plt.savefig("co2_emissions_over_time_Rl.png")
    plt.show()
    
    # Calculate the average CO2 emissions over all steps
    average_co2 = np.mean(co2_emissions)  # Calculate the mean of the CO2 emissions
    print(f"Average CO2 Emission over {len(co2_emissions)} steps: {average_co2:.2f} grams")

# Run the test simulation
if __name__ == "__main__":
    run_test_simulation(model_path="traffic_model.keras")
