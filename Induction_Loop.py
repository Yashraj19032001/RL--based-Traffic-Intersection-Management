import traci
import numpy as np
import sumolib
import os
import matplotlib.pyplot as plt

# Change to "sumo-gui" if you want the GUI
sumo_cmd = [
    "sumo-gui",
    "-c",
    "/Users/yashraj/Library/CloudStorage/OneDrive-TechnischeHochschuleIngolstadt/THI/Academics/Sem 3/General Elective/Smart Mobility/My Paper/New/RL--based-Traffic-Intersection-Management/Main_Simulation.sumocfg"
]

def get_waiting_times():
    waiting_times = []
    for lane in traci.lane.getIDList():
        wt = traci.lane.getWaitingTime(lane)
        waiting_times.append(wt)
    return waiting_times

def get_total_co2_emission_at_traffic_light():
    vehicle_ids = traci.vehicle.getIDList()
    total_co2 = 0
    for veh in vehicle_ids:
        speed = traci.vehicle.getSpeed(veh)
        if speed < 0.1:  # Consider the vehicle as 'stopped' if speed < 0.1 m/s
            total_co2 += traci.vehicle.getCO2Emission(veh)
    return total_co2

#-------------------------------------------------------------------
# NEW FUNCTION FOR THROUGHPUT MEASUREMENT
#-------------------------------------------------------------------
def measure_throughput():
    """
    Returns the number of vehicles that have arrived/completed their journey
    during this step. This indicates how many vehicles successfully
    passed through the network or reached their destination.
    """
    return traci.simulation.getArrivedNumber()

def track_vehicles_passed():
    vehicles_passed = 0
    for lane in traci.lane.getIDList():
        lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
        vehicles_passed += len(lane_vehicles)
    return vehicles_passed

def control_traffic_lights_by_loops():
    induction_loops = {
        "myLoop1": "North-South",
        "myLoop8": "East-West",
        "myLoop3": "West-East",
        "myLoop5": "South-North"
    }
    detection_threshold = 10
    for loop_id, direction in induction_loops.items():
        time_since_detection = traci.inductionloop.getTimeSinceDetection(loop_id)
        if time_since_detection > detection_threshold:
            if direction == "North-South":
                traci.trafficlight.setPhase("J1", 0)
            elif direction == "East-West":
                traci.trafficlight.setPhase("J1", 1)
            elif direction == "West-East":
                traci.trafficlight.setPhase("J1", 2)
            elif direction == "South-North":
                traci.trafficlight.setPhase("J1", 3)

def run_induction_loop_simulation():
    traci.start(sumo_cmd)

    # Lists to store data for each step
    all_waiting_times = []
    co2_emissions = []
    throughput_list = []  # NEW: vehicles that finished their routes each step

    vehicles_passed_total = 0
    total_co2 = 0
    total_waiting_time = 0
    total_delay = 0
    phase_delays = {0: 0, 1: 0, 2: 0, 3: 0}

    steps = 1000
    for step in range(steps):
        traci.simulationStep()

        waiting_times = get_waiting_times()
        all_waiting_times.append(waiting_times)

        step_total_waiting = sum(waiting_times)
        total_waiting_time += step_total_waiting

        step_co2 = get_total_co2_emission_at_traffic_light()
        co2_emissions.append(step_co2)
        total_co2 += step_co2

        # OPTIONAL: track how many vehicles pass through each lane
        vehicles_passed_step = track_vehicles_passed()
        vehicles_passed_total += vehicles_passed_step

        #-------------------------------------------------------------------
        # THROUGHPUT VIA getArrivedNumber (vehicles finishing their route)
        #-------------------------------------------------------------------
        arrived_vehicles = measure_throughput()
        throughput_list.append(arrived_vehicles)

        # Induction loop-based control
        control_traffic_lights_by_loops()

    np.savetxt("induction_loop_waiting_times.csv", all_waiting_times, delimiter=",")
    np.savetxt("induction_loop_co2_emissions.csv", co2_emissions, delimiter=",")
    np.savetxt("induction_loop_throughput.csv", throughput_list, delimiter=",")

    print(f"Total vehicles passed (lastStepVehicleIDs): {vehicles_passed_total}")
    print(f"Total CO2 emissions generated: {total_co2} g")
    print(f"Total waiting time at the signal: {total_waiting_time} seconds")
    print(f"Total delay caused by the traffic light: {total_delay} seconds")
    print(f"Delay by phase: {phase_delays}")

    traci.close()

    # Plot CO2 emissions
    plt.figure()
    plt.plot(range(1, len(co2_emissions) + 1), co2_emissions, marker='o', markersize=2, alpha=0.7)
    plt.title('CO2 Emissions Over Time (Stopped Vehicles)')
    plt.xlabel('Simulation Steps')
    plt.ylabel('CO2 Emission (g)')
    plt.grid()
    plt.savefig("co2_emissions_over_time.png")
    plt.show()

    # Plot Throughput
    plt.figure()
    plt.plot(range(1, len(throughput_list) + 1), throughput_list, marker='x', color='green', markersize=4, alpha=0.7)
    plt.title('Throughput (Arrived Vehicles) Over Time')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Arrived Vehicles')
    plt.grid()
    plt.savefig("throughput_over_time_induction.png")
    plt.show()

if __name__ == "__main__":
    run_induction_loop_simulation()
