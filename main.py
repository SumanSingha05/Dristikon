import cv2
from ultralytics import YOLO
import pyttsx3
import sqlite3
import time
from datetime import datetime
import subprocess
import sys

model = YOLO('yolov8n.pt')
conn = sqlite3.connect('drishti_memory.db')
c = conn.cursor()

try:
    c.execute("SELECT image_blob FROM memory LIMIT 1")
except sqlite3.OperationalError:
    c.execute("DROP TABLE IF EXISTS memory")

c.execute('''CREATE TABLE IF NOT EXISTS memory 
             (item TEXT, timestamp TEXT, surroundings TEXT, image_blob BLOB)''')
c.execute('''CREATE TABLE IF NOT EXISTS reminders 
             (msg TEXT, remind_time TEXT, status TEXT)''')
conn.commit()

memory_tracker = {}

def speak(text):
    print(f"AI: {text}")
    script = "import sys, pyttsx3; engine = pyttsx3.init(); engine.say(sys.argv[1]); engine.runAndWait()"
    creation_flags = 0x08000000 if sys.platform == "win32" else 0
    subprocess.Popen([sys.executable, '-c', script, text], creationflags=creation_flags)

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

def set_reminder():
    reminder_msg = input("What should I remind you about? ").strip()
    remind_time_str = input("When should I remind you (e.g., 01:15 AM or 1.15am)? ").strip().lower()
    
    try:
        t_str = remind_time_str.replace('.', ':')
        t_str = t_str.replace(' ', '')
        if 'am' in t_str or 'pm' in t_str:
            period = 'am' if 'am' in t_str else 'pm'
            time_part = t_str.replace(period, '')
            if ':' not in time_part:
                h = time_part
                m = "00"
            else:
                h, m = time_part.split(':')
            
            final_time = f"{int(h):02d}:{int(m):02d} {period.upper()}"
            
            c.execute("INSERT INTO reminders VALUES (?, ?, ?)", (reminder_msg, final_time, 'pending'))
            conn.commit()
            speak(f"Got it! I will remind you to {reminder_msg} at {final_time}.")
        else:
            speak("I didn't quite catch the time format. Please use something like 1:15 AM.")
    except Exception as e:
        speak(f"Sorry, I couldn't set that reminder. Error: {str(e)}")

def check_reminders():
    current_time = datetime.now().strftime("%I:%M %p")
    c.execute("SELECT rowid, msg FROM reminders WHERE remind_time=? AND status='pending'", (current_time,))
    pending = c.fetchall()
    
    for rowid, msg in pending:
        speak(f"Hey user, please {msg}. It's your reminder time, got it?")
        c.execute("UPDATE reminders SET status='done' WHERE rowid=?", (rowid,))
        conn.commit()

cap = cv2.VideoCapture(0)
speak("Drishti Setu system initialized. Passive scanning active.")

last_reminder_check = 0

try:
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
                        
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        c.execute("INSERT INTO memory VALUES (?, ?, ?, ?)", (label, timestamp, context, img_bytes))
                        conn.commit()
                        memory_tracker[label] = current_time
                        print(f"Log Updated: Saved image of {label} near {context}")

        cv2.imshow("DrishtiSetu - AI Vision Hub", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'): search_memory()
        elif key == ord('r'): set_reminder()
        
        current_time_sec = time.time()
        if (current_time_sec - last_reminder_check) > 1.0:
            check_reminders()
            last_reminder_check = current_time_sec

finally:
    print("AI: Shutting down systems...")
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
