import traci
import numpy as np
import sumolib
import os
import matplotlib.pyplot as plt

# Set up the SUMO simulation parameters
# Change to "sumo-gui" if you want to run it  GUI
sumo_cmd = ["sumo-gui", "-c", "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 3/General Elective/Smart Mobility/My Paper/New/RL--based-Traffic-Intersection-Management/Main_Simulation.sumocfg"]  # Adjust the path to your SUMO config file

def get_waiting_times():
    waiting_times = []
    for lane in traci.lane.getIDList():  # Iterate over all lanes
        waiting_time = traci.lane.getWaitingTime(lane)  # Get waiting time for each lane
        waiting_times.append(waiting_time)
    return waiting_times
  
def get_total_co2_emission_at_traffic_light():
    # Get all vehicle IDs in the simulation
    vehicle_ids = traci.vehicle.getIDList()  
    total_co2 = 0  # Initialize total CO2 emissions

    # Check all vehicles for CO2 emission if their speed is below threshold (0.1 m/s)
    for veh in vehicle_ids:
        # Get the vehicle's current speed
        speed = traci.vehicle.getSpeed(veh)

        # If the vehicle is stopped (speed is below 0.1 m/s), accumulate its CO2 emission
        if speed < 0.1:  # Consider the vehicle as stopped if speed is less than 0.1 m/s
            # Get the CO2 emission of the vehicle (this depends on the vehicle's activity)
            co2_emission = traci.vehicle.getCO2Emission(veh)
            total_co2 += co2_emission  # Add the emission to the total

    return total_co2


# Function to track number of vehicles that passed through the junction
def track_vehicles_passed():
    vehicles_passed = 0
    for lane in traci.lane.getIDList():  # Iterate through each lane
        # Check how many vehicles have passed through the lane
        lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)  # Get vehicles in the current step
        vehicles_passed += len(lane_vehicles)  # Count the vehicles passed in this lane
    return vehicles_passed

# Function to control traffic lights based on induction loop data
def control_traffic_lights_by_loops():
    # Define induction loop IDs for different lanes
    induction_loops = {
        "myLoop1": "North-South",
        "myLoop8": "East-West",
        "myLoop3": "West-East",
        "myLoop5": "South-North"
    }
    
    # Define a threshold for time since last detection (in seconds)
    detection_threshold = 10  # If a vehicle has been detected less than this time, assume it's waiting
    
    # Check the time since detection for each loop and control traffic lights
    for loop_id, direction in induction_loops.items():
        time_since_detection = traci.inductionloop.getTimeSinceDetection(loop_id)
        
        # If time since detection exceeds the threshold, switch the light to green for that direction
        if time_since_detection > detection_threshold:
            
            # Change the traffic light phase for the relevant direction
            if direction == "North-South":
                traci.trafficlight.setPhase("J1", 0)  # Set North-South green
            elif direction == "East-West":
                traci.trafficlight.setPhase("J1", 1)  # Set East-West green
            elif direction == "West-East":
                traci.trafficlight.setPhase("J1", 2)  # Set West-East green
            elif direction == "South-North":
                traci.trafficlight.setPhase("J1", 3)  # Set South-North green

def run_induction_loop_simulation():
    # Connect to the SUMO simulation
    traci.start(sumo_cmd)

    # List to store waiting times and CO2 emissions from each simulation step
    all_waiting_times = []
    co2_emissions = []
    vehicles_passed = 0  # Initialize the counter for passed vehicles
    total_co2 = 0  # Initialize total CO2 emissions across all steps
    total_waiting_time = 0  # Initialize total waiting time for all steps
    total_delay = 0  # Initialize total delay
    phase_delays = {0: 0, 1: 0, 2: 0, 3: 0}  # Initialize delay for each phase
    
    # Run the simulation for a specified number of steps or until finished
    for step in range(1000):  # Adjust the number of steps as needed
        traci.simulationStep()  # Perform one simulation step
        
       
        # # Accumulate total waiting time
        # total_waiting_time += waiting_time_step
        waiting_times = get_waiting_times()
        all_waiting_times.append(waiting_times)
       
        # Accumulate total waiting time
        total_waiting_time += sum(waiting_times)  # Sum the waiting times for all lanes in this step
        # Collect CO2 emissions for stopped vehicles at the intersection
        step_co2 = get_total_co2_emission_at_traffic_light()
        
        total_co2 += step_co2  # Add the CO2 emissions for the current step
        co2_emissions.append(step_co2)
        
        # Track number of vehicles passed through the junction
        vehicles_passed += track_vehicles_passed()

        # Control traffic lights based on induction loop data
        control_traffic_lights_by_loops()

    # Save the collected waiting times into a file for later comparison
    np.savetxt("induction_loop_waiting_times.csv", all_waiting_times, delimiter=",")
    
    # Save the collected CO2 emissions into a file for later comparison
    np.savetxt("induction_loop_co2_emissions.csv", co2_emissions, delimiter=",")
    
    # Print the total number of vehicles that passed through the junction
    print(f"Total number of vehicles passed through the junction: {vehicles_passed}")
    
    # Print the total CO2 emissions generated during the entire simulation
    print(f"Total CO2 emissions generated: {total_co2} g")
    
    # Print the total waiting time at the traffic signal
    print(f"Total waiting time at the signal: {total_waiting_time} seconds")
    
    # Print the total delay caused by the traffic lights
    print(f"Total delay caused by the traffic light: {total_delay} seconds")
    
    # Print delay for each traffic light phase
    print(f"Delay by phase: {phase_delays}")

    # Close the simulation
    traci.close()

    # Plot the CO2 emissions over time
    plt.figure()
    plt.plot(range(1, len(co2_emissions) + 1), co2_emissions, marker='o', markersize=2, alpha=0.7)
    plt.title('CO2 Emissions Over Time (Stopped Vehicles)')
    plt.xlabel('Simulation Steps')
    plt.ylabel('CO2 Emission (g)')
    plt.grid()
    plt.savefig("co2_emissions_over_time.png")
    plt.show()

if __name__ == "__main__":
    run_induction_loop_simulation()
