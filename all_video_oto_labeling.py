import cv2
import numpy as np
import os
from pathlib import Path
import zipfile
import shutil

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
      


def frame_capture(file):
    Path("all_obj_data/"+file+"/obj_train_data/").mkdir(parents=True, exist_ok=True)
    
    #open('all_obj_data/'+file+'train.txt', 'w').close()
         
    cap = cv2.VideoCapture(file)
    
    
    count = 0
    try:
        while True:
            
            success,frame = cap.read()
            cv2.imwrite("all_obj_data/"+file+"/obj_train_data/frame_%#06d.PNG" % count, frame)     # save frame as JPEG file    
           
            
            print('Read a new frame: ', success)
            with open("all_obj_data/"+file+"/train.txt", 'a') as f:
                        f.write('data/obj_train_data/frame_%#06d.PNG' % count)
                        f.write('\n')
                        f.close()
            
            #frame = cv2.flip(frame,1)
            frame = cv2.resize(frame,(608,608),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
           
            frame_blob = cv2.dnn.blobFromImage(frame, 0.00392, (608,608),(0, 0, 0),swapRB=True, crop=False)
        
            labels = ["person","car"]
        
               
            colors = ["0,0,255","0,0,255","255,0,0","255,255,0","0,255,0"]
            colors = [np.array(color.split(",")).astype("int") for color in colors]
            colors = np.array(colors)
            colors = np.tile(colors,(18,1))
        
        
            model = cv2.dnn.readNetFromDarknet("obj.cfg","obj.weights")
        
            layers = model.getLayerNames()
            output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
            
            model.setInput(frame_blob)
            
        
            
            detection_layers = model.forward(output_layer)
        
        
            ############## NON-MAXIMUM SUPPRESSION - OPERATION 1 ###################
            
            ids_list = []
            boxes_list = []
            confidences_list = []
            bounding_box=[]
            
            ############################ END OF OPERATION 1 ########################
            
            for detection_layer in detection_layers:
                for object_detection in detection_layer:
                    
                    scores = object_detection[5:]
                    predicted_id = np.argmax(scores)
                    confidence = scores[predicted_id]
                    
                    if confidence > 0.5:
                        
                        label = labels[predicted_id]
                        bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
                        (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                        
                         
                        start_x = int(box_center_x - (box_width/2))
                        start_y = int(box_center_y - (box_height/2))
                        
                       
                        
                        ############## NON-MAXIMUM SUPPRESSION - OPERATION 2 ###################
                        
                        ids_list.append(predicted_id)
                        confidences_list.append(float(confidence))
                        boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                        
                        ############################ END OF OPERATION 2 ########################
                        
              
             
                        
            ############## NON-MAXIMUM SUPPRESSION - OPERATION 3 ###################
                        
            max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.6)
            
            for max_id in max_ids:
             
                max_class_id = max_id[0]
                box = boxes_list[max_class_id]
                
                start_x = box[0] 
                start_y = box[1] 
                box_width = box[2] 
                box_height = box[3] 
                predicted_id = ids_list[max_class_id]
                label = labels[predicted_id]
                confidence = confidences_list[max_class_id]
                
                
    
               
              
            ############################ END OF OPERATION 3 ########################
                        
                end_x = start_x + box_width
                end_y = start_y + box_height
                
                print(end_x)
                print(end_y)
                print(start_x)
                print(start_y)
                print(frame.shape)
                 
                box_color = colors[predicted_id]
                box_color = [int(each) for each in box_color]
                
            
                        
                        
                label = "{}: {:.2f}%".format(label, confidence*100)
                #print("predicted object {}".format(label))
                 
                print(label)
                        
                cv2.rectangle(frame, (start_x,start_y),(end_x,end_y),box_color,2)
                #cv2.rectangle(frame, (start_x-1,start_y),(end_x+1,start_y-30),box_color,-1)
                cv2.putText(frame,label,(start_x,start_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            lines  = [] 
            for index in range(len(boxes_list)):
                if index in max_ids:
                    # yolo 1920 x 1080 resolution 
                    # example rectangle of points (100,100), (200,200) 
                    # yolo coordinates would be; 
                    
                    lines  = [int(ids_list[index]), float(box_center_x/frame_width), float(box_center_y/frame_height),float(end_x - start_x)/frame_width, float(end_y - start_y)/frame_height]  
                   
            with open("all_obj_data/"+file+"/obj_train_data/frame_%#06d.txt" % count,'w') as f:
                if len(boxes_list) == 0:
                    f.write(" ")
                else:
                  for line in lines:
                     f.write(str(line))
                     f.write(' ')
                  f.write('\n')
                  f.close()
                
                
            count += 1
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            cv2.imshow("Detector",frame)
            
    except Exception as e:
        print(str(e))
    
    cap.release()
    cv2.destroyAllWindows()
    
    save_path = 'all_obj_data/'+file
    file_name_1 = "obj.data"
    file_name_2 = "obj.names"
    
    shutil.copy(file_name_1, save_path, follow_symlinks=True)
    shutil.copy(file_name_2, save_path, follow_symlinks=True)
    
    zipf = zipfile.ZipFile(save_path+'.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(save_path, zipf)
    zipf.close()  
 

Path("all_obj_data/").mkdir(parents=True, exist_ok=True)

for file in os.listdir("videolar"):
    if file.endswith(".avi"):
        print(os.path.join("videolar/", file))
        print(file)
        print("videolar/"+file)
        frame_capture("videolar/"+file)

