import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Para la nave hay que usar el l, pero para cerezales mejor el x
model = YOLO('yolov8x.pt') # yolovl does not detect sheep if they are too far away
video_name = 'video19'
shepherd = 'robot'
#video_path = f"videos_Alexis/{video_name}.mp4"
video_path = f"ambasaguas/{video_name}.mp4"

cap = cv2.VideoCapture(video_path)

df = pd.DataFrame(columns=['frame', 'shepherd', 'frame_bboxes', 'min_distance', 'max_distance', 'avg_distance', 'furthest_sheep_distance'])

target_classes = ['dog', 'sheep', 'motorcycle', 'boat', 'cow', 'elephant', 'horse'] #'bird', 'zebra'
class_names = model.names
target_class_indices = [idx for idx, name in class_names.items() if name in target_classes]

fps = 30
#frame_interval = 5 # for speed estimation

# Custom function to visualize detections
def plot_results(frame, results, target_class_indices):
    frame_bboxes = []

    for result in results:
        for box in result.boxes:
            if int(box.cls) in target_class_indices:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_name = class_names[int(box.cls)]

                if class_name == 'motorcycle' or class_name == 'boat': #or class_name == 'zebra'
                    class_name = 'dog'

                if class_name == 'cow' or class_name == 'elephant' or class_name == 'horse': # bird
                    class_name = 'sheep'

                bbox = [x1, y1, x2, y2]
                if box.id:
                    print("BOX ID")
                    bbox_id = int(box.id)
                    frame_bboxes.append((bbox_id, class_name, bbox))

                    label = f"{class_name} id:{bbox_id}"

                    if class_name == 'sheep':
                        color = (232, 210, 44)  # Blue 
                    elif class_name == 'dog':
                        color = (52, 52, 235)  # Red
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    font_scale = 0.5 
                    font_thickness = 1
                    ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), font_thickness)

    return frame, frame_bboxes


# Function to calculate the middle point of the herd
def get_herd_center(frame_bboxes):
    sheep_centers = []

    for _, class_name, bbox in frame_bboxes:
        if class_name == 'sheep':
            x1, y1, x2, y2 = bbox
            center_x = (x1+x2)/2
            center_y = (y1+y2)/2
            sheep_centers.append((center_x, center_y))


    if sheep_centers:
        avg_center_x = np.mean([center[0] for center in sheep_centers])
        avg_center_y = np.mean([center[1] for center in sheep_centers])
        return (avg_center_x, avg_center_y)
    else:
        return None, None # No sheep found in the frame
    

# Function to calculate the distance between the furthest sheep
def get_distance_furthest_sheep(frame_bboxes):
    sheep_centers = []
    max_distance = 0

    for _, class_name, bbox in frame_bboxes:
        if class_name == 'sheep':
            x1, y1, x2, y2 = bbox
            center_x = (x1+x2)/2
            center_y = (y1+y2)/2
            sheep_centers.append((center_x, center_y))

    if sheep_centers:
        for i in range(len(sheep_centers)):
            for j in range(i + 1, len(sheep_centers)):
                distance = np.sqrt((sheep_centers[i][0] - sheep_centers[j][0]) ** 2 + (sheep_centers[i][1] - sheep_centers[j][1]) ** 2)
                if distance > max_distance:
                    max_distance = distance
        return max_distance
    else:
        return None # No sheep found in the frame
    

def closest_dog_to_herd(frame_bboxes):
    dog_centers = []

    for id, class_name, bbox in frame_bboxes: # Esto se podria juntar a la funcion de arriba
        if class_name == 'dog':
            x1, y1, x2, y2 = bbox
            center_x = (x1+x2)/2
            center_y = (y1+y2)/2
            dog_centers.append((id, center_x, center_y))

    if dog_centers:
        herd_center_x, herd_center_y = get_herd_center(frame_bboxes)
        if herd_center_x is not None:
            min_distance = float('inf')
            min_distance_id = None

            for id, center_x, center_y in dog_centers: # Get closest dog to herd_center
                distance = np.sqrt((center_x - herd_center_x) ** 2 + (center_y - herd_center_y) ** 2)
                
                if distance < min_distance:
                    min_distance = distance
                    min_distance_id = id

            return min_distance_id
        else:
            return None # No sheep found in the frame
    else:
        return None # No dogs found in the frame



def get_bbox_by_id(frame_bboxes, object_id):
    for bbox_id, class_name, bbox  in frame_bboxes:
        if bbox_id == object_id:
            return class_name, bbox
    return None, None  # id not found



# Calculates the distance of each sheep to the dog
def calculate_distances(frame_bboxes):
    distances = []
    
    dog_id = closest_dog_to_herd(frame_bboxes)
    if dog_id is not None:
        _, dog_bbox = get_bbox_by_id(frame_bboxes, dog_id)

        if dog_bbox is not None:

            for _, class_name, bbox in frame_bboxes:
                if class_name == 'sheep':
                    distance = np.sqrt((dog_bbox[0] - bbox[0])**2 + (dog_bbox[1] - bbox[1])**2)
                    distances.append(distance)
            return distances
        else:
            return None
    else:
        return None   
    


frame_number = 0
previous_positions = {}

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        # Filter results to only include 'dog' and 'sheep'
        filtered_boxes = []
        for result in results:
            filtered_boxes.extend([box for box in result.boxes if int(box.cls) in target_class_indices])

        results[0].boxes = filtered_boxes
        annotated_frame, frame_bboxes = plot_results(frame, results, target_class_indices)
        distances = calculate_distances(frame_bboxes)
        furthest_sheep_distance = get_distance_furthest_sheep(frame_bboxes)

        
        # Append the data for this frame to the DataFrame
        df.loc[len(df)] = {
            'frame': frame_number,
            'shepherd': shepherd,
            'frame_bboxes': frame_bboxes,
            'min_distance': min(distances) if distances else None,
            'max_distance': max(distances) if distances else None,
            'avg_distance': np.mean(distances) if distances else None,
            'furthest_sheep_distance': furthest_sheep_distance,
        }

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_number += 1

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

df.to_csv(f'data/{video_name}.csv', index=False)
