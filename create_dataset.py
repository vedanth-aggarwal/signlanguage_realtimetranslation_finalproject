import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import pickle # library to save datasets, models

#Iterate all frams, images and extract landmarks and save data to file 

# 3 OBJECTS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hands detector object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
NUM_KEYPOINTS = 21  # Number of keypoints per hand
'''
L
M
N
NO
O
P
Q
R
S
T
U
V
W
X
Y
YES
'''
data = []
labels = [] # categories 
for dir_ in os.listdir(DATA_DIR):
        original_length = len(data)
        count = 0
    #if dir_ in 'ABCDEFGHIJKLMNOPQRSTUVWXYZYES':
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []

            # Load and process the image
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:  # Considering only the first hand detected
                    # Get wrist landmark position
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y

                    for i in range(len(hand_landmarks.landmark)):
                        # Calculate relative coordinates for each keypoint
                        rel_x = hand_landmarks.landmark[i].x - wrist_x
                        rel_y = hand_landmarks.landmark[i].y - wrist_y

                        # Calculate absolute relative coordinates
                        abs_rel_x = abs(rel_x)
                        abs_rel_y = rel_y

                        data_aux.append(abs_rel_x)
                        data_aux.append(abs_rel_y)
                        if type(abs_rel_x) == type(rel_y) == float:
                             count+=1
                    # Fill in missing keypoints with zeros
                    #while len(data_aux) < NUM_KEYPOINTS * 2:
                     #   data_aux.append(0.0)

                
                data.append(data_aux)
                labels.append(dir_)
        print(dir_," : ",original_length-len(data)," : ",count)
        

# write binary
f = open('data.pickle','wb')
pickle.dump({'data':data,'labels':labels},f)
f.close()





""" mp_drawing.draw_landmarks(
                    img_rgb, # image to draw
                    hand_landmarks, # model output 
                    mp_hands.HAND_CONNECTIONS, # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())"""
            
        #plt.figure()
        #plt.imshow(img_rgb) # requires rgb image 

#plt.show()

    