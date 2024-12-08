# induction_loop_simulation.py

import traci
import numpy as np
import sumolib
import os

# Set up the SUMO simulation parameters
sumo_binary = "sumo-gui"  # You can change to "sumo" if you want to run it without GUI
sumo_cmd = ["sumo", "-c", "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 3/General Elective/Smart Mobility/My Paper/New/RL--based-Traffic-Intersection-Management/Main_Simulation.sumocfg"]  # Adjust the path to your SUMO config file

# Function to get the waiting times from SUMO
def get_waiting_times():
    waiting_times = []
    for lane in traci.lane.getIDList():  # Iterate over all lanes
        waiting_time = traci.lane.getWaitingTime(lane)  # Get waiting time for each lane
        waiting_times.append(waiting_time)
    return waiting_times

def run_induction_loop_simulation():
    # Connect to the SUMO simulation
    traci.start(sumo_cmd)

    # List to store waiting times from each simulation step
    all_waiting_times = []

    # Run the simulation for a specified number of steps or until finished
    for step in range(10000):  # Adjust the number of steps as needed
        traci.simulationStep()  # Perform one simulation step
        
        # Collect waiting times after each step
        waiting_times = get_waiting_times()
        all_waiting_times.append(waiting_times)
        
        # Example logic for Induction Loop system (you can adjust)
        for lane in traci.lane.getIDList():
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
            if vehicle_count > 0:
                traci.trafficlight.setPhase("J1", 0)  # Green phase for this intersection

    # Save the collected waiting times into a file for later comparison
    np.savetxt("induction_loop_waiting_times.csv", all_waiting_times, delimiter=",")
    
    # Close the simulation
    traci.close()

if __name__ == "__main__":
    run_induction_loop_simulation()
