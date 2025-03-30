from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from googletrans import Translator

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

# location categorized
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

def draw_bounding_boxes(frame, results):
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    img_height, img_width = frame.shape[0], frame.shape[1]

    for i in range(len(labels)):
        x1, y1, x2, y2, conf = coords[i]
        x1, y1, x2, y2 = int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_height)
        label = model.names[int(labels[i])]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

def provide_audio_feedback(results, img_width, img_height):
    translator = Translator()
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    detected_objects = defaultdict(lambda: {'count': 0, 'locations': defaultdict(int)})

    for i in range(len(labels)):
        label = model.names[int(labels[i])]
        x1, y1, x2, y2 = int(coords[i][0] * img_width), int(coords[i][1] * img_height), int(coords[i][2] * img_width), int(coords[i][3] * img_height)
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

    # Translate the feedback to Italian/or any supported language
    translated_feedback = translator.translate(feedback, src='en', dest='it').text

    print(translated_feedback)

    tts = gTTS(translated_feedback, lang='it')
    tts.save('audio_feedback.mp3')
    audio = AudioSegment.from_mp3('audio_feedback.mp3')
    play(audio)
    os.remove('audio_feedback.mp3')

# Capturing video from webcam
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
