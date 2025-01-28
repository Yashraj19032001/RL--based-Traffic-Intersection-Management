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
        num_vehicles = traci.lane.getLastStepVehicleNumber(lane)
        state.append(num_vehicles)

        waiting_time = traci.lane.getWaitingTime(lane)
        state.append(waiting_time)

        traffic_light_phase = traci.trafficlight.getPhase("J1")  # Junction_ID
        state.append(traffic_light_phase)

    return state

def apply_action(action):
    traffic_light_id = "J1"  # Junction_ID
    if action == 0:
        traci.trafficlight.setPhase(traffic_light_id, 0)
    elif action == 1:
        traci.trafficlight.setPhase(traffic_light_id, 1)
    elif action == 2:
        traci.trafficlight.setPhase(traffic_light_id, 2)
    elif action == 3:
        traci.trafficlight.setPhase(traffic_light_id, 3)

def save_waiting_times(waiting_times, filename="rl_waiting_times.csv"):
    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(waiting_times)
            print(f"Saving waiting times row: {waiting_times}")
    except Exception as e:
        print(f"Error saving waiting times: {e}")

def save_co2_emissions(co2_emissions, filename="rl_co2_emission.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(co2_emissions)

def get_total_co2_emission_at_traffic_light():
    vehicle_ids = traci.vehicle.getIDList()
    total_co2 = 0

    for veh in vehicle_ids:
        speed = traci.vehicle.getSpeed(veh)
        if speed < 0.01:  # Consider vehicle 'stopped'
            total_co2 += traci.vehicle.getCO2Emission(veh)
    return total_co2

# -----------------------------------------------------------------
# NEW FUNCTION FOR THROUGHPUT MEASUREMENT
# -----------------------------------------------------------------
def measure_throughput():
    """
    Returns the number of vehicles that have arrived/completed their journey
    during this step. This is an indicator of how many vehicles successfully
    passed through the network or reached their destination.
    """
    return traci.simulation.getArrivedNumber()

def save_throughput(throughputs, filename="rl_throughput.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(throughputs)

# Run the RL simulation to test the model
def run_test_simulation(model_path="traffic_model.keras"):
    sumo_binary = "sumo"  
    sumo_config = "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 3/General Elective/Smart Mobility/My Paper/New/RL--based-Traffic-Intersection-Management/Main_Simulation.sumocfg"

    traci.start([sumo_binary, "-c", sumo_config, "--no-step-log", "true", "--log", "false"])

    # Load the saved model
    model = load_model(model_path)

    steps = 1001  # total simulation steps for testing
    batch_size = 100
    waiting_times_per_batch = []
    co2_emissions = []
    throughput_list = []  # Store throughput (arrived vehicles) per step
    
    current_batch_waiting_times = []
    current_batch_co2_emissions = []
    current_batch_throughputs = []

    for step in range(steps):
        traci.simulationStep()  # Advance the simulation

        # Current state from SUMO
        state = np.array([get_traffic_state()])

        # Model predicts best action
        action_values = model.predict(state)
        action = np.argmax(action_values)

        # Apply predicted action
        apply_action(action)

        # Calculate total waiting time
        total_waiting_time = sum(get_traffic_state()[1::3])
        current_batch_waiting_times.append(total_waiting_time)

        # Calculate CO2 for stopped vehicles
        total_co2 = get_total_co2_emission_at_traffic_light()
        co2_emissions.append(total_co2)
        current_batch_co2_emissions.append(total_co2)

        # MEASURE THROUGHPUT
        arrived_vehicles = measure_throughput()
        throughput_list.append(arrived_vehicles)
        current_batch_throughputs.append(arrived_vehicles)

        # If batch is complete, compute averages
        if (step + 1) % batch_size == 0:
            avg_waiting_time = np.mean(current_batch_waiting_times)
            waiting_times_per_batch.append(avg_waiting_time)
            save_waiting_times([avg_waiting_time])
            
            avg_co2_emissions = np.mean(current_batch_co2_emissions)
            save_co2_emissions([avg_co2_emissions])
            
            # Save throughput (sum or average for the batch)
            batch_throughput = np.sum(current_batch_throughputs)
            save_throughput([batch_throughput])

            # Reset for next batch
            current_batch_waiting_times = []
            current_batch_co2_emissions = []
            current_batch_throughputs = []

        # Print progress
        print(f"Step {step + 1}/{steps} - Action: {action} - CO2 Emission: {total_co2} - Throughput: {arrived_vehicles}")

    traci.close()

    # Plot average waiting time per batch
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

    # Plot throughput over time
    plt.figure()
    plt.plot(range(1, steps + 1), throughput_list, marker='x', color='green', markersize=3, alpha=0.7)
    plt.title('Throughput (Arrived Vehicles) Over Time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Arrived Vehicles')
    plt.grid()
    plt.savefig("throughput_over_time_Rl.png")

    plt.show()
    
    # Calculate and print average CO2 emissions over all steps
    average_co2 = np.mean(co2_emissions)
    print(f"Average CO2 Emission over {len(co2_emissions)} steps: {average_co2:.2f} grams")

# Run the test simulation
if __name__ == "__main__":
    run_test_simulation(model_path="traffic_model.keras")
