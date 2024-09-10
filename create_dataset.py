import os
import pickle
import mediapipe as mp
import cv2

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Check if the data directory exists
if not os.path.exists(DATA_DIR):
    raise ValueError(f"The directory {DATA_DIR} does not exist!")

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    print(f"Processing directory: {dir_}")
    
    if not os.path.isdir(dir_path):
        print(f"Skipping non-directory file: {dir_path}")
        continue

    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        print(f"Processing image: {img_path}")
        
        if not os.path.isfile(img_full_path):
            print(f"Image file not found: {img_full_path}")
            continue
        
        # Load image
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Failed to load image: {img_full_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print(f"No hand landmarks detected for image: {img_path}, skipping.")
            continue

        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        data.append(data_aux)
        labels.append(dir_)

# Save the data
print("Saving dataset to data.pickle...")
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset creation complete!")
