**extract_data**: Python program for extracting data from the recorded videos and saving it into a CSV file.

**data_processing**: Jupyert Notebook that takes the extracted data and calculates the parameters related to speed.

**yolo_visualize**: Python program for visualizing the YOLOv8 detections of the videos.

**plot_generator**: program for generating plots from the combined data CSVs.


**data**: folder with the CSVs for all the videos. Inside **processed_data** are the dataframes after adding the estimated speeds of all objects. "combined_data" has the data from the preliminary evaluation, and "final_evaluation" contains the data from the videos of the final evaluation.

Each row of the dataframe represents a frame of the video.
The columns of the dataframes are:

- **frame**: id of the frame starting from 0.
- **frame_bboxes**: list of all the detections in the frame. Each detection stores detection_id, class_name and bounding box(x1,y1,x2,y2).
- **distances**: list of distances of every individual sheep to the detected dog. It is calculated using the center of the bbox as reference.
- **min_distance**: minimum sheep-dog distance.
- **max_distance**: maximum sheep-dog distance.
- **avg_distance**: mean sheep-dog distance.
- **furthest_sheep_distance**: maximum distance between 2 sheep.
- **speeds**: list of the estimated speed of every sheep.
- **dog_speed**: the estimated speed of the dog.
- **min_speed**: the speed of the slowest detected sheep in the herd.
- **max_speed**: the speed of the fastest detected sheep in the herd.
- **avg_speed**: the mean speed of the herd.

