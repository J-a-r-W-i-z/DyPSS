"""
Run this command to convert screenshot frames to video:
ffmpeg -y -framerate 5 -i signal_frames/frame_%04d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -r 30 -pix_fmt yuv420p emergency_vehicle_tracking.mp4
"""

import traci
from time import sleep
import os
import requests
import base64
import cv2
import numpy as np
import threading

# CCTV camera settings
CAM_ROI =  {
    "cluster_2683490416_2683507646_2683507647_2684658305_#2more" : [1546, 0, 3192, 1644],
    "cluster_2705363613_2705372495_297516017_3138462219_#1more" : [770, 0, 2414, 1664],
    "cluster_297516018_3138462223_7274261863_7274261864" : [770, 0, 2414, 1664],
    "cluster_297133110_4952251210" : [1546, 0, 3192, 1644],
}

# Directory to save the screenshots
vehicle_frames_dir = "vehicle_frames/"
if not os.path.exists(vehicle_frames_dir):
    os.makedirs(vehicle_frames_dir)
signal_frames_dir = "signal_frames/"
if not os.path.exists(signal_frames_dir):
    os.makedirs(signal_frames_dir)
evacuation_frames_dir = "evacuation_frames/"
if not os.path.exists(evacuation_frames_dir):
    os.makedirs(evacuation_frames_dir)
etans_frames_dir = "etans_frames/"
if not os.path.exists(etans_frames_dir):
    os.makedirs(etans_frames_dir)

# Start SUMO simulation with TraCI
sumoBinary = "sumo-gui"  # Use SUMO GUI to capture screenshots
sumoCmd = [sumoBinary, "-c", "osm.sumocfg", "--start", "--step-length", "0.1"]  # Automatically start SUMO
traci.start(sumoCmd)


# Define the emergency vehicle ID, route, and camera settings
emergency_vehicle_id = "emergency_vehicle"
route_id = "emergency_route"
zoom_level = 2500  # Adjust zoom level for tracking the vehicle

# Define the locations (edges) for route A to B
start_edge = "314582543"  # Replace with actual start edge ID
end_edge = "1012009845#1"    # Replace with actual end edge ID

# Find the complete route using SUMO's internal routing
route = traci.simulation.findRoute(fromEdge=start_edge, toEdge=end_edge)

# Extract the list of edges in the calculated route
full_route_edges = [edge for edge in route.edges]

print(f"Calculated route: {full_route_edges}")

# Set Traffic Factor for the simulation
traci.simulation.setScale(3)

# Fog controller IP address and port
fog_controller_ip = "127.0.0.1"
fog_controller_port = 8080

# Function to read image data from a file
def read_image_data(image_path, tls_id):

    # Read the image using OpenCV
    old_image = cv2.imread(image_path)

    # Crop the old image to make it square
    h, w = old_image.shape[:2]
    y1, y2, x1, x2 = 0, h, 0, w
    if h > w:
        y1 = (h-w) // 2
        y2 = y1 + w
    else:
        x1 = (w-h) // 2
        x2 = x1 + h
    cropped = old_image[y1:y2, x1:x2]

    # Overwrite the image_path with the cropped image
    cv2.imwrite(image_path, cropped)

    # Crop the old image to the region of interest (ROI) according to the camera settings
    x1_roi, y1_roi, x2_roi, y2_roi = CAM_ROI[tls_id]
    image = old_image[y1_roi:y2_roi, x1_roi:x2_roi]

    # Resize the image to a fixed size for processing
    image = cv2.resize(image, (640, 640))

    # Convert the image to bytes
    _, img_encoded = cv2.imencode('.jpg', image)

    # Convert the image bytes to base64 encoding
    image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return image_base64

# Helper function for visualization
def create_image_from_text(text, filename):
    # Create a black image (1500x500 pixels)
    image = np.zeros((500, 1000, 3), dtype=np.uint8)
    
    # Set font type, scale, color, and thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 255, 255)  # White color
    thickness = 3
    
    # Get the size of the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate the position to center the text
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    
    # Put the text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness)
    
    # Save the image to a file
    cv2.imwrite(filename, image)


# Variables to store the traffic light details that helps in actuation
original_program = None
upcoming_tls = None
actuated = False
counter = 0
num_steps = 0

def actuate_traffic_signal():
    global original_program, upcoming_tls, actuated, counter, num_steps
    print("Checking...")
    # Get the details of the upcoming traffic lights for the emergency vehicle
    tls_info = traci.vehicle.getNextTLS(emergency_vehicle_id)
    if not tls_info:
        print("No traffic lights ahead.")
        # Exit the thread if the vehicle has passed all traffic lights
        return
    
    tls_id, tls_index, distance, state = tls_info[0]

    if upcoming_tls != tls_id:
        actuated = False
        # Reset the tls to default state
        if original_program:
            traci.trafficlight.setProgram(upcoming_tls, original_program)
        upcoming_tls = tls_id
        print(f"Upcoming traffic light: {upcoming_tls}")
        original_program = traci.trafficlight.getProgram(upcoming_tls)
    
    # Get the speed limit of the current lane
    speed_limit = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(emergency_vehicle_id))

    # Update the camera position to focus on the upcoming traffic light
    tls_position = traci.junction.getPosition(tls_id)
    traci.gui.setOffset(viewID="View #0", x=tls_position[0], y=tls_position[1])
    
    # Get a screenshot of the upcoming traffic light
    screenshot_filename = f"{signal_frames_dir}frame_{counter:04d}.png"
    traci.gui.screenshot(viewID="View #0", filename=screenshot_filename)
    traci.simulationStep()
    num_steps += 1
    print(f"Step: {num_steps}")
    
    # Calculate the estimated time to reach the traffic light
    eta = distance / speed_limit
    file_name = f"{etans_frames_dir}frame_{counter:04d}.png"
    text = f"ETANS = {eta:.2f}s"
    create_image_from_text(text, file_name)

    # Read the image (CCTV input emulation)
    image = read_image_data(screenshot_filename, tls_id)

    # Request evacuation time from the fog controller
    response = requests.post(f"http://{fog_controller_ip}:{fog_controller_port}/evacuation_time", json={"image": image, "tls_id": tls_id})
    # Get evacuation time from the response
    evacuation_time = response.json()["evacuation_time"]
    file_name = f"{evacuation_frames_dir}frame_{counter:04d}.png"
    text = f"Evacuation Time = {evacuation_time}s"
    create_image_from_text(text, file_name)

    print(f"Evacuation time: {evacuation_time}")
    # evacuation_time = 10  # Placeholder value for demonstration
    
    # If the evacuation time is greater than the ETA, change the traffic light to green
    if evacuation_time > eta and not actuated:
        print(f"Actuating traffic signal {tls_id} to green.")
        
        # with traci_lock:
        # Get the current traffic light phase
        current_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        current_phase = current_program[0].phases[tls_index]
        lanes_controlled = traci.trafficlight.getControlledLanes(tls_id)

        print("Fetched current phase and lanes controlled.")
        
        # Change the phase state: set green for lanes on the road and red for others
        green_state = list(current_phase.state)

        # Loop through all lanes controlled by the traffic light
        for i, lane in enumerate(lanes_controlled):
            # Check if the lane's edge belongs to the road on which the vehicle is approaching
            lane_edge_id = lane.split('_')[0]  # Extract the edge ID from the lane ID
            if lane_edge_id in full_route_edges:
                # Set the signal to green for lanes on the relevant road
                green_state[i] = 'G'
            else:
                # Set the signal to red for all other lanes
                green_state[i] = 'r'

        # Create the new traffic light state with green for the target road and red for others
        new_state = ''.join(green_state)
        
        # Set the new traffic light state for the next phase
        traci.trafficlight.setRedYellowGreenState(tls_id, new_state)
        actuated = True
        print(f"Signal at {tls_id} changed to green for lane.")
    counter += 1

    
# Main function to control the simulation
if __name__ == "__main__":
    # Run simulation for random number of steps to initialize the simulation
    initalize = 1200
    for _ in range(initalize):
        traci.simulationStep()

    # Add the emergency vehicle route
    traci.route.add(routeID=route_id, edges=full_route_edges)

    # Deploy the emergency vehicle at the start of the simulation
    traci.vehicle.add(vehID=emergency_vehicle_id, typeID="emergency_veh", routeID=route_id)
    # traci.vehicle.setColor(emergency_vehicle_id, (255, 0, 0, 255))  # Red color for visibility

    # Move to the emergency vehicle's initial position and zoom in
    initial_position = traci.vehicle.getPosition(emergency_vehicle_id)
    traci.gui.setOffset(viewID="View #0", x=initial_position[0], y=initial_position[1])
    traci.gui.setZoom(viewID="View #0", zoom=zoom_level)

    step = 0
    # Main simulation loop
    while True: 
        try:
            # Update camera position to follow the emergency vehicle
            vehicle_position = traci.vehicle.getPosition(emergency_vehicle_id)
            traci.gui.setOffset(viewID="View #0", x=vehicle_position[0], y=vehicle_position[1])
            screenshot_filename = f"{vehicle_frames_dir}frame_{step:04d}.png"
            traci.gui.screenshot(viewID="View #0", filename=screenshot_filename)
            traci.simulationStep()

            # Read the image using OpenCV
            old_image = cv2.imread(screenshot_filename)

            # Crop the old image to make it square
            h, w = old_image.shape[:2]
            y1, y2, x1, x2 = 0, h, 0, w
            if h > w:
                y1 = (h-w) // 2
                y2 = y1 + w
            else:
                x1 = (w-h) // 2
                x2 = x1 + h
            old_image = old_image[y1:y2, x1:x2]

            # Overwrite the image_path with the cropped image
            cv2.imwrite(screenshot_filename, old_image)

            num_steps += 1
            print(f"Step: {num_steps}")

        except traci.TraCIException:
            print("Emergency vehicle is no longer in the simulation.")
            break

        step += 1

        actuate_traffic_signal()

    # Close the traffic signal actuation thread
    # signal_thread.join()

    # Close TraCI after simulation
    traci.close()

    print("Number of steps:", num_steps)   