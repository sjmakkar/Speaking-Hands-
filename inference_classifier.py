import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import StringVar
from PIL import Image, ImageTk
import time
import pyttsx3  # Import the text-to-speech library

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Label dictionary for all alphabets (replace with your full labels)
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z labels

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

# Cooldown and consistency thresholds
cooldown_time = 2  # Seconds between detections
last_detected_time = 0
previous_character = None
detection_consistency = 0
consistency_threshold = 5  # Number of consistent detections to confirm the character

# Function to process frame and predict sign
def process_frame():
    global last_detected_time, previous_character, detection_consistency

    ret, frame = cap.read()
    if not ret:
        return None, None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    predicted_character = None

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []
        H, W, _ = frame.shape

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the coordinates
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        # Predict character if valid data length
        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

        # Draw the landmarks and bounding box
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        # Draw bounding box and predicted character
        if len(x_) > 0:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            if predicted_character:
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Ensure cooldown between consecutive detections
    current_time = time.time()
    if predicted_character and (current_time - last_detected_time > cooldown_time):
        # Confirm the character only if it is consistent over multiple frames
        if predicted_character == previous_character:
            detection_consistency += 1
            if detection_consistency >= consistency_threshold:
                # Reset detection consistency and return the character
                detection_consistency = 0
                last_detected_time = current_time  # Update last detection time
                return frame, predicted_character
        else:
            # If a different character is detected, reset consistency
            previous_character = predicted_character
            detection_consistency = 1

    return frame, None

# Function to update the video feed and handle predictions
def update():
    frame, character = process_frame()
    if frame is not None:
        # Convert the frame to a format suitable for Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        video_label.config(image=img)
        video_label.image = img

    if character:
        # Append the detected character to the sentence
        sentence.set(sentence.get() + character)

    # Call the update function after 10ms
    root.after(10, update)

# Function to add space between words and speak the word
def add_space():
    # Speak the sentence formed so far when a space is added
    engine.say(sentence.get())
    engine.runAndWait()

    # Add a space after the spoken word
    sentence.set(sentence.get() + " ")

# Function to reset the sentence
def reset_sentence():
    sentence.set("")

# Initialize camera and Tkinter window
cap = cv2.VideoCapture(0)
root = tk.Tk()
root.title("Indian Sign Language Detection")

# Create GUI elements
video_label = tk.Label(root)
video_label.pack()

sentence = StringVar()
sentence_box = tk.Label(root, textvariable=sentence, font=("Arial", 18), bg="white", width=40, height=2)
sentence_box.pack(pady=10)

# Buttons for space and reset
space_button = tk.Button(root, text="Add Space", command=add_space)
space_button.pack(side=tk.LEFT, padx=10)

reset_button = tk.Button(root, text="Reset Sentence", command=reset_sentence)
reset_button.pack(side=tk.RIGHT, padx=10)

# Quit button to close the app
quit_button = tk.Button(root, text="Quit", command=root.quit)
quit_button.pack(pady=10)

# Start the video capture and processing
update()

# Start Tkinter main loop
root.mainloop()

# Release resources
cap.release()
