# Initialize the YOLO model
model =YOLO('yolov8n.pt')

def load_image(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)
    return img

def detect_objects(img):
    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Perform detection
    results = model(img_rgb)
    return results

def draw_bounding_boxes(img, results):
    # Extract bounding boxes, labels, and confidence scores
    img_height, img_width = img.shape[0], img.shape[1]
    
    for *xyxy, conf, cls in results.xyxy[0]:  # results.xyxy is a tensor of shape (N, 6)
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        label = model.names[int(cls)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Add label and confidence score
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return img

# Example usage
img_path = 'image.jpg'
img = load_image(img_path)
results = detect_objects(img)
img_with_boxes = draw_bounding_boxes(img, results)

# Display the image with bounding boxes
cv2.imshow('Detected Objects', img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
