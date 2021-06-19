#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import os
import getpass as gp
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


# ##  Creating Training Data

# In[ ]:


#Loading harcascade model for face recognition
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Function to Crop face using haarcascade
def  face_crop(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_img, 1.3, 2)
    
    if faces == ():
        return None
    
    for (x,y,width,hieght) in faces:
        cropped_face = image[y:y+hieght, x:x+width]
        
    return cropped_face

 

def CollectSampleImages(object_name):
    
    try:
        os.mkdir("faces/{}".format(object_name))   
    except FileExistsError:
        print("Object with name {} already exists!! Try different name!!".format(object_name))
        return
    
    #Initialising
    cam = cv2.VideoCapture(1)
    count = 0
       
    while True:
        ret, img = cam.read()
        if face_crop(img) is not None:
            count += 1
            face = cv2.resize(face_crop(img),(250,250))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            #Save file in in ./faces/ directory with unique name
            file_path = "./faces/{}/".format(object_name) + str(count)+ ".jpg"
            cv2.imwrite(file_path, face)

            # Put count on images and display live count
            cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
            cv2.imshow('Cropped face' , face)

        else:
            print("Face not found !!")

        if cv2.waitKey(1) == 32 or count == 100: #Space bar is pressed
            break;

    #Release Camera and close window 
    cam.release()
    cv2.destroyAllWindows()
    if count == 100:
        print("Samples Collected Successfully in ./faces/{}/ directory".format(object_name))
    return
     


# In[ ]:


CollectSampleImages("aditi")


# In[ ]:


CollectSampleImages("adarsh")


# ## Creating Model

# In[ ]:



def trainmodel(object_name):

    data_path = "./faces/{}/".format(object_name)

    #Including only those files which are in ./faces/user/ path
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # Create arrays for training data and labels
    Training_Data, Labels = [], []
    # Open training images in our datapath
    # Create a numpy array for training data
    for i, files in enumerate(onlyfiles):
        image_path = data_path + files
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)

    # Initialize facial recognizer
    # model = cv2.face.createLBPHFaceRecognizer()
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
    # pip install opencv-contrib-python
    # model = cv2.createLBPHFaceRecognizer()

    model = cv2.face_LBPHFaceRecognizer.create()
    # Let's train our model 
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained sucessefully")
    return model

#List storing models of all objects to be trained
object_models, object_list = [], ["Aditi", "Adarsh"]
object_models.append(trainmodel("aditi"))
object_models.append(trainmodel("adarsh"))


# ## Creating function to Send Whatsapp Message, Email and apply terraform code

# In[ ]:


import smtplib
import pywhatkit as pwk
from python_terraform import *
from pprint import pprint 

def sendmail(sender_mail, reciever_mail,message, sender_password):
    print("Sending mail...")
    try:
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        print(s.login(sender_mail, sender_password))
        s.sendmail(sender_mail, reciever_mail, message)
        s.quit()
        print("Successfully sent email")    
    except Exception as e:
        print("Error: unable to send email : {}".format(e)) 
        
def whatsapp(phone_num, message):
    pwk.sendwhatmsg_instantly(phone_num,message)
    
def terraform_apply(workingDir):
    try:
        #Creating instance and volume using terraform
        tf = Terraform(working_dir=workingDir)
        tf.plan(no_color=IsFlagged, refresh=False, capture_output=True)
        #approve = {"auto-approve": True}
        tf.init()
        pprint(tf.plan())
        pprint(tf.apply(skip_plan=True))
        
    except Exception as e:
        print("Some Error occured!! {}".format(e))
        
    
    
    


# ## Function to Recognize faces

# In[ ]:


def recognize_face(object_models, object_list, face, image):
    identity = ""
    for i,model in enumerate(object_models):
        result = model.predict(face)
        max_confidence = 0
        if result[1] < 500:
            confidence = int( 100 * (1 - (result[1])/400) )
            display_string = str(confidence) + '% Confident it is {}'.format(object_list[i])
            cv2.putText(image, display_string, (150, 120+(i*30)), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
            cv2.imshow('Face Recognition', image)
        if confidence > 85:
            if confidence > max_confidence:
                max_confidence = confidence
                identity = object_list[i]
            cv2.putText(image, "Hey {}".format(identity), (250, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image)
        else:
            cv2.putText(image, "Trying to Recognize..", (200, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
            cv2.imshow('Face Recognition', image)
       
    return identity    


# ## Testing Model

# In[ ]:




face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_detector(image,size=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces == ():
            return image,[]
        
    faces_detected = []   
    for (x,y,width,hieght) in faces:
        cv2.rectangle(image,(x,y),(x+width,y+hieght),(0,255,255),2)
        cropped_face = img[y:y+hieght, x:x+width]
        cropped_face = cv2.resize(cropped_face, (250, 250))
        faces_detected.append(cropped_face)    
    return image, faces_detected



cam = cv2.VideoCapture(1)

task_status_a = False
task_status_b = False
exitKeyPressed = False
password = gp.getpass("Enter Sender's mail Password: ")

while True:
    ret, img = cam.read()
    image, faces_detected = face_detector(img)
    try:
        for i,face in enumerate(faces_detected): 
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Pass face to prediction models
            object_name = recognize_face(object_models, object_list,face_gray, image)
            #print(object_name)
            
            if not task_status_a and object_name == "Aditi":
                sendmail("sender@gmail.com", "reciever@gmail.com", "This is face of Aditi",password)
                whatsapp("+91998******", "Hi... Model recognized me!!") 
                task_status_a = True
            
            elif not task_status_b and object_name == "Adarsh":
                #Applying terraform code
                terraform_apply('./terraform-ws/aws_ec2_ebs/')
                task_status_b =True        
            else:
                cv2.putText(image, "I dont know, who r u", (230, 470), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
                cv2.imshow('Face Recognition', image)
                
            if cv2.waitKey(1) == 32: #32 is the Space bar
                exitKeyPressed = True
                break  
    except:
        pass
    
    finally:
        if task_status_a and task_status_b:
            print("ALL task Completed Successfully... You can exit now by pressing space key!")
        if exitKeyPressed or cv2.waitKey(1) == 32:
            break 

    
        
cam.release()
cv2.destroyAllWindows()     



# In[ ]:




