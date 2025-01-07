import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')  # Replace 'best.pt' with the actual model you want to use.

# Load the image
image_path = '3rd.png'
image = cv2.imread(image_path)

# Resize the image to 640x640
image = cv2.resize(image, (640, 640))

# Define the four coordinates of the quadrilateral
quad_coords = np.array([[280, 360], [340, 360], [420, 640],[360, 640]])  # Update these values as needed

# Function to check if a point is inside a polygon
def point_in_polygon(x, y, polygon):
    result = cv2.pointPolygonTest(polygon, (x, y), False)
    return result >= 0

# Perform object detection
results = model(image)

# Initialize counters for the specific object types
bus_count = 0
car_count = 0
motorcycle_count = 0

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

        # Check if the detected object is within the specified quadrilateral
        if point_in_polygon(x_center, y_center, quad_coords):
            print(f'Object detected within the specified region: {model.names[int(class_id)]}')
            print(f'Center coordinates: ({x_center}, {y_center})')
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

# Draw the quadrilateral on the image
cv2.polylines(image, [quad_coords], isClosed=True, color=(255, 0, 0), thickness=2)

# Resize the image for display (optional)
image = cv2.resize(image, (800, 600))
cv2.imshow('YOLOv8 Detection', image)
cv2.waitKey(0)
# Save the imge
cv2.imwrite('output.png', image)
cv2.destroyAllWindows()
