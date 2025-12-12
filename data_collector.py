import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open CSV file for appending
csv_file = open('gesture_data.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press o/f/p/t/r/v/l to record OPEN/FIST/PEACE/THUMBS_UP/ROCK/VICTORY/LOVE or quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            cv2.putText(frame, 'Press o/f/p/t/r/v/l to record OPEN/FIST/PEACE/THUMBS_UP/ROCK/VICTORY/LOVE or quit.', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('o'):
                csv_writer.writerow(landmarks + ['OPEN'])
                print("[SAVED] OPEN")
            elif key == ord('f'):
                csv_writer.writerow(landmarks + ['FIST'])
                print("[SAVED] FIST")
            elif key == ord('p'):
                csv_writer.writerow(landmarks + ['PEACE'])
                print("[SAVED] PEACE")
            elif key == ord('t'):
                csv_writer.writerow(landmarks + ['THUMBS_UP'])
                print("[SAVED] THUMBS_UP")
            elif key == ord('r'):
                csv_writer.writerow(landmarks + ['ROCK'])
                print("[SAVED] ROCK")  # Rock (metal sign)
            elif key == ord('v'):
                csv_writer.writerow(landmarks + ['VICTORY'])
                print("[SAVED] VICTORY") # Victory (peace with palm facing you)
            elif key == ord('q'):
                cap.release()
                csv_file.close()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow('Hand Data Collector', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
