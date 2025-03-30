# this code is based on real time example through webcam

# Load custom YOLO model
# in my case the custom model was trained on manual dataset and the exported version destroyed from my previous laptop :(
from ultralytics import YOLO
model = YOLO('yolov8n.pt') # refer to size of this version

# function for localizing objects in 9 different directions
def get_location_category(x1, y1, x2, y2, img_width, img_height):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    if center_x < img_width / 3:
        if center_y < img_height / 3:
            return "top-left"
        elif center_y < 2 * img_height / 3:
            return "left"
        else:
            return "bottom-left"
    elif center_x < 2 * img_width / 3:
        if center_y < img_height / 3:
            return "up"
        elif center_y < 2 * img_height / 3:
            return "center"
        else:
            return "down"
    else:
        if center_y < img_height / 3:
            return "top-right"
        elif center_y < 2 * img_height / 3:
            return "right"
        else:
            return "bottom-right"
          
# bounding boxes around the objects
def draw_bounding_boxes(frame, results):
    labels = results[0].boxes.cls
    coords = results[0].boxes.xyxy
    confs = results[0].boxes.conf
    img_height, img_width = frame.shape[0], frame.shape[1]

    for i in range(len(labels)):
        x1, y1, x2, y2 = int(coords[i][0]), int(coords[i][1]), int(coords[i][2]), int(coords[i][3])
        label = model.names[int(labels[i])]
        conf = confs[i]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# real time audio feedback
def provide_audio_feedback(results, img_width, img_height):
    labels = results[0].boxes.cls
    coords = results[0].boxes.xyxy
    detected_objects = defaultdict(lambda: {'count': 0, 'locations': defaultdict(int)})

    for i in range(len(labels)):
        label = model.names[int(labels[i])]
        x1, y1, x2, y2 = int(coords[i][0]), int(coords[i][1]), int(coords[i][2]), int(coords[i][3])
        location_category = get_location_category(x1, y1, x2, y2, img_width, img_height)
        detected_objects[label]['count'] += 1
        detected_objects[label]['locations'][location_category] += 1

    if detected_objects:
        feedback_parts = []
        for label, info in detected_objects.items():
            count = info['count']
            if count == 1:
                location = next(iter(info['locations']))
                feedback_parts.append(f"one {label} at the {location}")
            else:
                locations = [f"{loc_count} at the {loc}" for loc, loc_count in info['locations'].items()]
                feedback_parts.append(f"{count} {label}s: " + ", ".join(locations))
        feedback = "I see: " + ", ".join(feedback_parts)
    else:
        feedback = "No objects detected."

    tts = gTTS(feedback, lang='en')
    tts.save('audio_feedback.mp3')
    audio = AudioSegment.from_mp3('audio_feedback.mp3')
    play(audio)
    os.remove('audio_feedback.mp3')

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    frame_with_boxes = draw_bounding_boxes(frame, results)
    
    img_height, img_width = frame.shape[0], frame.shape[1]
    provide_audio_feedback(results, img_width, img_height)

    cv2.imshow('Webcam', frame_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
