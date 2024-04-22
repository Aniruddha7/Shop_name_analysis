# Shop nameboard detection and recognition using in Yolov5 object detection model

##Object detection of single or multiple objects and inferencing it on Google colab. Since Colab provides free GPUs such as Nvidia Tesla T4, we can instantly use these resources for training and testing the deep learning models. Here are the steps for implementing this model for detecting nameboards.
1. Create custom dataset of cars(~400) and annotate them using tools such as VGG image annotator or LabelImg

2. Divide the dataset into train test format

3. Calculation of anchor boxes according to dataset

4. Zip the prepared data and upload it to Google Colab

5. Load the datset and train the model

6. Inference the model as the final step

Step1:
Here the downloaded images are annotated using VGG/LanelImg/Roboflow annotator tool and are put usder the folder final_dataset.zip. As we annotate each images, we get the corresponding .txt file that has respective annotations. The number of classes required are mentioned in data.yaml file.

Step2:
In this step the data into train, valid and test images. The dataset has 377 images and is split into 70%-20%-10% resepectively.

Step3:
Anchor boxes are calculated as each image undergoas annotation using K-means clustering and saves it as a txt file.

Step4:
Open Colab and mount the Google drive and start training the model. The model is trained for first 50 epochs with pretrained Yolov5L weights on MS COCO dataset and the custom weightfile is saved.Afterwards the model is again trained for next 75 epochs with finetuning and is cached. 

Inferencing: The trained weights can be inferenced on images, video mp4 files or streamed on YouTUbe video links using detect.py file. 

