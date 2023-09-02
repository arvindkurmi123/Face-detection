# Recognise faces using some classification falgorithm like - logistic, knn, svm etc


# 1. load the trainging data (numpy arrays of all the persons)
#     X - values are stored in the numpy arrays
#     y - values we need to assign for each person

# 2. Read a video stream using openCV
# 3. Extract faces out of it
# 4. Use KNN to find the prediction of face (int)
# 5. map the prediction id to name of the User 
# 6. Display the predictions onn the screen - bounding box and NameError

import cv2
import numpy as np
import os

##### KNN Code ######

def distance(v1,v2):
    #Eucledian
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]

        # compute the distance form test point
        d = distance(test,ix)
        dist.append([d,iy])

    # sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]

    # retrieve only the labels
    labels  = np.array(dk)[:,-1]

    # get frequencies and corresponding label
    output = np.unique(labels,return_counts=True)

    # find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

##############################################################

# Init Camera

cap = cv2.VideoCapture(0)

# Face Detection 
face_cascade = cv2.CascadeClassifier("openCV\haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

face_data = []
labels = []

class_id = 0 # Labels for the given file
names = {}  # Mapping btw id - name

# Data preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # create a mapping btw class_id and name

        names[class_id] = fx[:-4]
        print("Loaded "+fx)

        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # create Labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

# Testing 

while True:

    ret,frame = cap.read()

    if ret == False:
        continue

    # gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    # faces = sorted(faces,key=lambda f: f[2]*f[3])

    face_section = frame[0:1,0:1]
    # pic the last face (because it is the largest face acc to area(f[2]*f[3]))
    for face in faces:
        x,y,w,h = face
        # extract (crop out the required face ): reason of interest
        offset  = 10
        # by default the axis of frame [Y,X]
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        # predicted label (out)
        out = knn(trainset,face_section.flatten())

        # Display on the screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        

    cv2.imshow("FACES",frame)
    

    key_pressed  = cv2.waitKey(1) & 0xFF 
    if key_pressed == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()