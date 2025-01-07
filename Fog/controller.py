import http.server
import socketserver
import threading
import requests
import json
import time
import sys
import random
import numpy as np

MEM_USAGE_TIMEOUT = 1
MAX_TRY = 3
FOG_NODES = [
    {"ip": "127.0.0.1:5000"}
    # {"ip": "192.168.0.105:5000"},
    # {"ip": "192.168.0.104:5000", "username": "pi3", "password": "raspberry1"},
    # {"ip": "192.168.0.102:5000", "username": "pi1", "password": "raspberry1"}
    # Add more Fog Nodes as needed
]

# Quadrilateral coordinates for object detection ROI
DETECTION_ROI = {
    "cluster_2683490416_2683507646_2683507647_2684658305_#2more" : [[100, 340], [620, 450], [600, 530],[80, 420]],
    "cluster_2705363613_2705372495_297516017_3138462219_#1more" : [[280, 360], [320, 360], [320, 630],[280, 630]],
    "cluster_297516018_3138462223_7274261863_7274261864" : [[280, 360], [340, 360], [420, 640],[360, 640]],
    "cluster_297133110_4952251210" : [[60, 310], [640, 0], [640, 50],[80, 360]]
}

# Data structure to store memory usage
memory_usage = {}
for i in range(len(FOG_NODES)):
    memory_usage[i]=0

# Create a lock
memory_usage_lock = threading.Lock()

def update_memory_usage():
    """
    Thread function to periodically update memory usage data for each fog node.
    """
    nodes = FOG_NODES
    random.shuffle(nodes)
    while True:
        # print("Getting Mem Usage from fog nodes")
        for i in range(len(nodes)):
            # Update memory_usage dictionary
            temp = get_memory_usage(nodes[i]["ip"])
            with memory_usage_lock:
                memory_usage[i] = temp
        # Update every time chosen randomly between 5 to 15 seconds
        time.sleep(random.randint(5, 15))


def get_memory_usage(fog_node_ip):
    """
    Get the memory usage of a fog node.

    Args:
        fog_node_ip (str): The IP address of the fog node.

    Returns:
        float: The memory usage as a float.
    """
    url = f"http://{fog_node_ip}/get_memory_usage"
    try:
        response = requests.get(url,  timeout=MEM_USAGE_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get('free_memory', 0)
        else:
            print("Error: Unexpected status code:", response.status_code)
            return 0
    except Exception as e:
        print(f"Error in getting memory usage, (node {fog_node_ip} not responding)")
        print(e)
        return 0
    

# Function to select a fog node with a probability proportional to free memory
def placement_algo_probabilistic():
    """
    Select a fog node with a probability proportional to free memory.

    Returns:
        str: The IP address of the selected fog node.
    """
    with memory_usage_lock:
        total_memory = sum(value for value in memory_usage.values())
        if total_memory == 0:
            return -1
        probabilities = [memory / total_memory for memory in memory_usage.values()]
        # print("Probabilities: ", probabilities)
    selected_index = random.choices(range(len(FOG_NODES)), probabilities)[0]
    return FOG_NODES[selected_index]["ip"]


# Handler for Fog Controller
class FogControllerHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        """
        Handle requests from clients.
        """
        # For client use
        if self.path == "/evacuation_time":
            print("Client requested...")
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            image_data = json.loads(post_data.decode('utf-8'))
            tls_id = image_data['tls_id']

            # Select the fog node for processing
            fog_node_ip = placement_algo_probabilistic()
            
            if fog_node_ip == -1:
                print("Error: No fog node available")
                self.send_response(500)
                self.end_headers()
                return
                
            print(f"Forwarding serivce execution request to Fog Node at IP- {fog_node_ip}")
            # Get roi for the image
            roi = DETECTION_ROI[tls_id]
            response = requests.post(f"http://{fog_node_ip}/service_execution", json={"image": image_data['image'], "roi": roi, "num_lanes": 2})
            if response.status_code == 200:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(response.content)
            else:
                self.send_response(500)
                self.end_headers()
                return


def run_fog_controller(port):
    """
    Run the Fog Controller on the specified port.

    Args:
        port (int): The port number to run the Fog Controller on.
    """
    with socketserver.ThreadingTCPServer(("", port), FogControllerHandler) as server:
        print(f"Fog Controller running on port {port}")
        server.serve_forever()


if __name__ == "__main__":

    if len(sys.argv) == 2:
        fog_controller_port = int(sys.argv[1])
    else:
        print("Error: Incorrect number of arguments.")
        print("Usage: python controller.py <port>")
        exit(1)

    # Start memory usage update thread
    memory_thread = threading.Thread(target=update_memory_usage)
    memory_thread.daemon = True
    memory_thread.start()

    fog_controller_thread = threading.Thread(target=run_fog_controller, args=(fog_controller_port,))
    fog_controller_thread.daemon = True
    fog_controller_thread.start()

    fog_controller_thread.join()
    memory_thread.join()