import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

from Test4 import is_autistic

# Define a set of colors for different parts
colors = {
    'left_eye': (255, 0, 0),    # Blue
    'right_eye': (0, 255, 0),   # Green
    'nose': (0, 0, 255),        # Red
    'lips': (255, 255, 0),      # Cyan
    'upper_face': (255, 0, 255),# Magenta
    'middle_face': (0, 255, 255),# Yellow
    'philtrum': (255, 165, 0)   # Orange
}

def extract_face_parts(image, keypoints):
    parts = {}
    parts_boxes = {}
    
    for part, (x, y) in keypoints.items():
        if part == 'left_eye':
            parts['left_eye'] = image[max(y-30, 0):y+30, max(x-30, 0):x+30]
            parts_boxes['left_eye'] = (max(x-30, 0), max(y-30, 0), x+30, y+30)
        elif part == 'right_eye':
            parts['right_eye'] = image[max(y-30, 0):y+30, max(x-30, 0):x+30]
            parts_boxes['right_eye'] = (max(x-30, 0), max(y-30, 0), x+30, y+30)
        elif part == 'nose':
            parts['nose'] = image[max(y-30, 0):y+30, max(x-30, 0):x+30]
            parts_boxes['nose'] = (max(x-30, 0), max(y-30, 0), x+30, y+30)
        elif part in ['mouth_left', 'mouth_right']:
            mx = (keypoints['mouth_left'][0] + keypoints['mouth_right'][0]) // 2
            my = (keypoints['mouth_left'][1] + keypoints['mouth_right'][1]) // 2
            parts['lips'] = image[max(my-20, 0):my+20, max(mx-40, 0):mx+40]
            parts_boxes['lips'] = (max(mx-40, 0), max(my-20, 0), mx+40, my+20)
    
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    
    upper_y = max(min(left_eye[1], right_eye[1]) - 50, 0)
    upper_x1 = max(min(left_eye[0], right_eye[0]) - 50, 0)
    upper_x2 = min(max(left_eye[0], right_eye[0]) + 50, image.shape[1])
    parts['upper_face'] = image[upper_y:min(left_eye[1], right_eye[1])+30, upper_x1:upper_x2]
    parts_boxes['upper_face'] = (upper_x1, upper_y, upper_x2, min(left_eye[1], right_eye[1])+30)

    middle_y = max(nose[1] - 40, 0)
    middle_x1 = max(nose[0] - 50, 0)
    middle_x2 = min(nose[0] + 50, image.shape[1])
    parts['middle_face'] = image[middle_y:nose[1]+40, middle_x1:middle_x2]
    parts_boxes['middle_face'] = (middle_x1, middle_y, middle_x2, nose[1]+40)

    philtrum_y = nose[1]
    philtrum_x1 = mouth_left[0]
    philtrum_x2 = mouth_right[0]
    parts['philtrum'] = image[max(philtrum_y-20, 0):philtrum_y+20, philtrum_x1:philtrum_x2]
    parts_boxes['philtrum'] = (philtrum_x1, max(philtrum_y-20, 0), philtrum_x2, philtrum_y+20)

    return parts, parts_boxes

def prepare_7_channel_input(part_image):
    if part_image.shape[2] == 3:  # Check if the image has 3 channels (RGB)
        part_image_64 = cv2.resize(part_image, (64, 64))  # Resize to model input size
        part_image_64 = part_image_64 / 255.0  # Normalize the image
        
        # Create a 7-channel image
        part_7_channel = np.zeros((64, 64, 7), dtype=np.float32)
        part_7_channel[:, :, :3] = part_image_64  # Assign the RGB channels
        
        return np.expand_dims(part_7_channel, axis=0)  # Add batch dimension
    else:
        raise ValueError("The input image should have 3 channels (RGB)")

def predict_facial_parts(image_path, model):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)

    if faces:
        keypoints = faces[0]['keypoints']
        parts, parts_boxes = extract_face_parts(image_rgb, keypoints)
        part_predictions = {}

        for part, img in parts.items():
            if img.size > 0:
                input_data = prepare_7_channel_input(img)
                prediction = model.predict(input_data)
                part_predictions[part] = prediction[0][0]  # Get the first value of the prediction
        
        # Draw bounding boxes and results on the image
        for part, box in parts_boxes.items():
            x1, y1, x2, y2 = box
            color = colors.get(part, (0, 255, 0))  # Default color is green if not specified
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            prediction_text = f"{part}: {part_predictions.get(part, 0):.2%}"
            cv2.putText(image, prediction_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Determine overall prediction
        overall_prediction = np.mean(list(part_predictions.values()))
        # overall_text = "Autistic" if overall_prediction >= 0.5 else "Non-Autistic"
        overall_color = (0, 0, 255) if overall_prediction >= 0.5 else (0, 255, 0)

        # Draw overall result
        # cv2.putText(image, f"Overall: {overall_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, overall_color, 3)
        # cv2.putText(image, f"Probability: {overall_prediction:.2%}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overall_color, 2)

        return image, part_predictions, overall_prediction
    else:
        print("No face detected.")
        return None, None, None

# Load the trained model
model_path = 'Face_Model.h5'
model = load_model(model_path)



# Provide the path to the image you want to test
image_path = r'Autisum_Face_Img\test\autistic\006.jpg'



# Predict each part and draw bounding boxes
image, part_predictions, overall_prediction = predict_facial_parts(image_path, model)
if part_predictions:
    for part, probability in part_predictions.items():
        print(f"{part}: {probability:.2%} likelihood of autism")

    # Display the image with bounding boxes and predictions
    cv2.imshow("Predictions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"No face detected in {image_path}.")

if is_autistic(image_path, model):
    print(f"The person in {image_path} is predicted to be autistic.")
else:
    print(f"The person in {image_path} is predicted not to be autistic.")

