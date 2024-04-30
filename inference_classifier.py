import cv2
import mediapipe as mp
import pickle 
import numpy as np

model_dict = pickle.load(open('./model.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hands detector object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3,max_num_hands=2)

#labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    #progress.text('Analysing....')
    data_aux = []
    x_ = []
    y_ = []

    try:
        ret, frame = cap.read()
    except:
        pass
   
    #if frame is None:
    #   continue  # Skip this iteration and try capturing the next frame

    H, W, _ = frame.shape
    #if not hand_detected:
     #   pass
        #canvas.image(frame, channels="BGR")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            if True :
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output 
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Get wrist landmark position
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            for i in range(len(hand_landmarks.landmark)):
                # Calculate relative coordinates for each keypoint
                x = hand_landmarks.landmark[i].x 
                y = hand_landmarks.landmark[i].y

                new_x = abs(hand_landmarks.landmark[i].x - wrist_x)
                new_y = hand_landmarks.landmark[i].y - wrist_y

                

                data_aux.append(new_x)
                data_aux.append(new_y)
                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20

            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = prediction[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_char, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    
    cv2.imshow('frame',frame)
    # wait 25 miliseconds between each frame
    cv2.waitKey(1)


cap.release() # release memory
cv2.destroyAllWindows()

