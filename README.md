# Yolo-auto-labeling-engine
openCV python- automatically labeling all the videos in the folder and prepares output suitable for Yolo CVAT as a zip file in a separate folder.

First of all, you should upload CVAT to the videos that you will do automatic annotation and create a task.

Before running:
The obj.data and obj.names files are organized according to your model.
The model folder is created for the weights and cfg files and the files are added.
The videos folder with the video files is created and the video files are added.


After running:
The all_obj_data folder is created. 
The videos folder is created in this all_obj_data folder. 
In this video folder, it gives a zip file output in CVAT yolo format for each video file.

You can upload the zip files to the tasks of the videos you previously created in CVAT with the option of upload annotations -> yolo 1.1 format.
