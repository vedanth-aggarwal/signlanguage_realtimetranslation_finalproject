''' DELTA AI FELLOWSHIP : FINAL PROJECT - SIGN LANGUAGE DETECTOR '''

######################### NECESSARY IMPORTS + API KEY #########################################################

import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import time
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = "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"

###################### LOADING MODEL AND MEDIAPIPE LANDMARKS #############################################

model_dict = pickle.load(open('./model.p','rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hands detector object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8,max_num_hands=1)

####################### SETTING PAGE LAYOUT ##############################################################

st.set_page_config(layout="wide")

st.title("SignCharm - Sign Language Prediction Project")

st.markdown("<hr>", unsafe_allow_html=True)

# Create a canvas to display video and landmarks
col1,col2 = st.columns(2)
with col1: 
    canvas = st.empty()
    frame_interval = st.slider("Temperature", min_value=0, max_value=30, step=1, value=15, format="Interval : %d", label_visibility='collapsed')
    
# Create a text area to display the detected characters
with col2:
    # Create a checkbox for live speech translation
    live_speech_toggle = st.checkbox("Live sign language to speech translation")
    # Create a checkbox for enabling hand landmarks on webcam display
    hand_landmarks_toggle = st.checkbox("Mediapipe hand landmarks on webcam display")

    prediction_text = st.empty()
    special_command = st.empty()
    refined_text = st.empty()

st.markdown("<hr>", unsafe_allow_html=True)

################################ OTHER INITIALIZATIONS ###################################################

cap = cv2.VideoCapture(0)

detected_characters = []
frame_count = 0

def text_to_speech(text):
   ''' Convert text entered directly to audio without needing to save file '''
   engine = pyttsx3.init()
   engine.say(text)
   engine.runAndWait()

hand_detected = False # Are user hands detected in webcam
display = True
############################# MAIN FRAMEWORK - SIGN LANGUAGE RECOGNITION LOGIC #################################

while True:
    data_aux = []
    x_ = []
    y_ = []

    try:
        ret, frame = cap.read()
    except:
        pass
   
    if frame is None:
        continue      # Skip this iteration and try capturing the next frame

    H, W, _ = frame.shape
    if hand_detected is False:
        canvas.image(frame, channels="BGR")
        #cv2.putText(frame,'No hand detected!', (200,100), cv2.FONT_HERSHEY_COMPLEX, 1.3, (45, 45, 255), 3, cv2.LINE_AA)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            if hand_landmarks_toggle and display: # If mediapipe drawings is enabled
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

            # Boundary box dimensions 
            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20

            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = prediction[0]

            if display:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_char, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame,'Program Stopped!', (200,50), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            # Display the webcam feed with detected landmarks on the canvas
            canvas.image(frame, channels="BGR")

            # Update detected characters and display in text area
            if frame_count % frame_interval == 0:
                if predicted_char == '_stop_':
                    special_command.error('Program stopped!')
                    display = False

                elif predicted_char == '_start_':
                    prediction_text.info("Predicted : " + "".join(detected_characters))
                    special_command.error('Program started!')
                    refined_text.success(f'Refined Text : ')
                    display = True

                elif display:
                    if predicted_char == '_delchar_':
                        letter = ""
                        if detected_characters != []:
                            letter = detected_characters.pop(-1)
                        else:
                            continue

                        prediction_text.info("Predicted : " + "".join(detected_characters))
                        if letter == ' ':
                            special_command.error(f'Letter deleted : *space*')
                        else:
                            special_command.error(f'Letter deleted : {letter}')

                    elif predicted_char == '_delword_':
                        if detected_characters != []:
                            if detected_characters.count('_') < 2:
                                word = "".join(detected_characters)
                                detected_characters.clear()
                            else:
                                word = ""
                                x = ""
                                while x != " ":
                                    word += x
                                    x = detected_characters.pop(-1)
                        prediction_text.info("Predicted : " + "".join(detected_characters))
                        special_command.error(f'Word deleted : {word}')
                    
                    elif predicted_char == '_clear_':
                        detected_characters.clear()
                        prediction_text.info("Predicted : " + "".join(detected_characters))
                        special_command.error('Text cleared.....')

                    elif predicted_char == '_refine_':
                        prediction_text.info("Predicted : " + "".join(detected_characters))
                        special_command.error('Text being refined.....')
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{'role':'user','content':f"This is a text generated using sign language recognition. Your job \
                            is to correctly punctuate it, ensure capitalizations, maintain proper gramatical structure, \
                                and sentence phrasing should be good. Do not enhance or change the meaning of the sentence \
                                    just return the sentence with the following changes.\
                                        Sentence : <{''.join(detected_characters).replace('_',' ')}>"}],
                                        temperature=0.7,stream=False
                            )
                        result = ''
                        for choice in response.choices:
                            result += choice.message.content
                        refined_text.success(f'Refined Text : {result}')
                        text_to_speech(result)

                    elif predicted_char == '_space_':
                        detected_characters.append(' ')
                        if live_speech_toggle:
                            if detected_characters.count(' ') > 1:
                                text_to_speech("".join(detected_characters[''.join(detected_characters).rindex(' '):]))
                            else:
                                text_to_speech("".join(detected_characters))
                        prediction_text.info("Predicted : " + "".join(detected_characters))
                        special_command.error(f'Added : *space*')
                        refined_text.success('Refined Text : ')
                    else:
                        if live_speech_toggle:
                            text_to_speech(predicted_char)
                        if len(predicted_char) > 1:
                            detected_characters.append(" ")
                        detected_characters.append(predicted_char)
                        prediction_text.info("Predicted : " + "".join(detected_characters))
                        special_command.error(f'Added : {predicted_char}')
                        refined_text.success('Refined Text : ')
                       
    else:
        hand_detected = False
        frame_count = 0
    
    cv2.waitKey(15)      
    frame_count += 1


cap.release()
cv2.destroyAllWindows()

#########################################################################################################