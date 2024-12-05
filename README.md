# pcb-detector
Project for my Bachelor's Thesis, trying to compare pcb-s and their corresponding schematics
Planning to use YOLO  
https://yolov8.com/  

some projects i have found with a similar purpose:  
https://github.com/s39674/Image2schematic - creates schematics from pictures of pcb-s  
https://github.com/VanillaHours/pcbDefectDetectionYOLO -pcb defects with yolov3  
https://www.nature.com/articles/s41598-022-16302-3 - pcb defects with yolo  
https://www.kaggle.com/datasets/akhatova/pcb-defects - train set for pcb defects  
https://universe.roboflow.com/component-detection-for-pcb/component-dzbbw  
https://universe.roboflow.com/pcbboard/pcb-board-component
  
Open source version of yolo for use  
https://github.com/WongKinYiu/YOLO?tab=readme-ov-file  
  
I aim to train my model on Ouman OÜ pcb-s and schematics
  
  
TODO:  
- Get exact timings of how fast the model can or should work.  
- Scale up the detection to use the full resolution for better results.  
- Try out ResNext101, to see how it compares with YOLO.  
- Make a list of components I need to detect for certain.  
- Inquire what the accuracy should be and how often can the needed component go past detection without being noted.  
- Write the theoretical part of the paper before 8th of december.  
