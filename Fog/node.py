import http.server
import socketserver
import threading
import json
import base64
import numpy as np
import sys
import cv2
import psutil
from ultralytics import YOLO

# Lock to synchronize access to the memory usage data
model_lock = threading.Lock()

# Load the YOLOv8 model 
model = YOLO('best.pt')

class FogNodeHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/get_memory_usage':
            try:
                print("Getting Mem usage...")
                mem = psutil.virtual_memory()
                free_memory =  mem.available

            except:
                print("Error in getting memory usage")
                self.send_response(500)
                self.end_headers()
                response = {'free_memory': None}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return

            try:
                # Send the response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'free_memory': free_memory}
                self.wfile.write(json.dumps(response).encode('utf-8'))

            except:
                print("Error in sending response. (controller went down possibly)")

    def do_POST(self):
        if self.path == "/service_execution":
            print("Executing Service")
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            image_data = json.loads(post_data.decode('utf-8'))
            roi = image_data['roi']
            print("Image received")

            # Convert the base64 image data to cv2 image
            image_bytes = base64.b64decode(image_data['image'])
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract the region of interest (ROI) coordinates and number of lanes
            num_lanes = image_data['num_lanes']

            evacuation_time = process_image(image, roi, num_lanes)
            print(f"Evacuation time: {evacuation_time}")

            response = {'evacuation_time': evacuation_time}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            

# Function to check if a point is inside a polygon
def point_in_polygon(x, y, polygon):
    result = cv2.pointPolygonTest(polygon, (x, y), False)
    return result >= 0

# Run YOLOv8 model and return object detection results
def process_image(image, roi, num_lanes):
    # Initialize counters for the specific object types
    bus_count = 0
    car_count = 0
    motorcycle_count = 0
    results = None
    with model_lock:
        results = model(image)

    # Extract the detected boxes, classes, and confidences
    # Extract the detected boxes, classes, and confidences
    for result in results:
        boxes = result.boxes.xyxy  # Get the bounding boxes in [x1, y1, x2, y2] format
        confidences = result.boxes.conf  # Get the confidence scores
        class_ids = result.boxes.cls  # Get the class IDs

        # Draw the bounding boxes and labels on the image
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  # Convert to integer
            label = f'{model.names[int(class_id)]}: {confidence:.2f}'
            color = (0, 255, 0)  # Bounding box color (green)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # Put the label text above the bounding box
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            print(f'Center coordinates: ({x_center}, {y_center})')

            quad_coords = np.array(roi)
            # Check if the detected object is within the specified quadrilateral
            if point_in_polygon(x_center, y_center, quad_coords):
                print(f'Object detected within the specified region: {model.names[int(class_id)]}')
                # Check if the detected class is one of the target classes
                class_name = model.names[int(class_id)]
                if class_name == 'Bus':
                    bus_count += 1
                elif class_name == 'Car':
                    car_count += 1
                elif class_name == 'Motor':
                    motorcycle_count += 1

    # Display the counts
    print(f'Number of buses in the specified region: {bus_count}')
    print(f'Number of cars in the specified region: {car_count}')
    print(f'Number of motorcycles in the specified region: {motorcycle_count}')

    # Calculate the evacuation time based on the object counts
    evacuation_time = (motorcycle_count*2 + car_count*3 + bus_count*4)/num_lanes
    return evacuation_time

def run_fog_node(port):
    with socketserver.ThreadingTCPServer(("", port), FogNodeHandler) as server:
        print("Fog Node running on port "+ str(port))
        server.serve_forever()

if __name__ == "__main__":
    fog_node_port = None
    if len(sys.argv) == 2:
       fog_node_port = int(sys.argv[1])
    else:
       print("Error: Incorrect number of arguments.")
       print("Usage: python node.py <port>")
       exit(1)
    fog_node_thread = threading.Thread(target=run_fog_node, args=(fog_node_port,))
    fog_node_thread.start()