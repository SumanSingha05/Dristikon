import cv2
from ultralytics import YOLO
import pyttsx3
import sqlite3
import time
from datetime import datetime

engine = pyttsx3.init()
model = YOLO('yolov8n.pt')
conn = sqlite3.connect('drishti_memory.db')
c = conn.cursor()

try:
    c.execute("SELECT image_blob FROM memory LIMIT 1")
except sqlite3.OperationalError:
    c.execute("DROP TABLE IF EXISTS memory")

c.execute('''CREATE TABLE IF NOT EXISTS memory 
             (item TEXT, timestamp TEXT, surroundings TEXT, image_blob BLOB)''')
conn.commit()

memory_tracker = {}

def speak(text):
    print(f"AI: {text}")
    engine.say(text)
    engine.runAndWait()

def search_memory():
    item_to_find = input("Enter the item you are looking for: ").lower()
    c.execute("SELECT timestamp, surroundings, image_blob FROM memory WHERE item=? ORDER BY timestamp DESC LIMIT 1", (item_to_find,))
    result = c.fetchone()
    
    if result:
        time_seen, place, img_blob = result
        speak(f"I last saw your {item_to_find} at {time_seen}. It was near the {place}.")
        
        ask_more = input(f"Would you like more details about this {item_to_find}? (y/n): ").lower()
        if ask_more == 'y':
            prompt = input("What would you like to know? (e.g., color, looks): ")
            speak("Analyzing image, please wait...")
            
            import moondream_utils as md_file
            description = md_file.describe_object(img_blob, prompt)
            speak(f"Based on what I saw: {description}")
    else:
        speak(f"I'm sorry, I haven't seen a {item_to_find} yet.")

cap = cv2.VideoCapture(0)
speak("Drishti Setu system initialized. Passive scanning active.")

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, conf=0.5, verbose=False)
    
    for r in results:
        potential_surroundings = [model.names[int(b.cls[0])] for b in r.boxes if model.names[int(b.cls[0])] in ['dining table', 'couch', 'bed', 'laptop', 'tv']]
        context = potential_surroundings[0] if potential_surroundings else "your current location"

        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            current_time = time.time()
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            if width > (frame.shape[1] * 0.5):
                cv2.putText(frame, "OBSTACLE ALERT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if label not in memory_tracker or (current_time - memory_tracker[label]) > 30:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    _, buffer = cv2.imencode('.jpg', crop)
                    img_bytes = buffer.tobytes()
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    c.execute("INSERT INTO memory VALUES (?, ?, ?, ?)", (label, timestamp, context, img_bytes))
                    conn.commit()
                    memory_tracker[label] = current_time
                    print(f"Log Updated: Saved image of {label} near {context}")

    cv2.imshow("DrishtiSetu - AI Vision Hub", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'): search_memory()

cap.release()
cv2.destroyAllWindows()
conn.close()
