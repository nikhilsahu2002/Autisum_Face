import cv2
import os
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

def extract_face_parts(image, keypoints):
    parts = {}
    for part, (x, y) in keypoints.items():
        if part == 'left_eye':
            parts['left_eye'] = image[y-30:y+30, x-30:x+30]
        elif part == 'right_eye':
            parts['right_eye'] = image[y-30:y+30, x-30:x+30]
        elif part == 'nose':
            parts['nose'] = image[y-30:y+30, x-30:x+30]
        elif part == 'mouth_left' or part == 'mouth_right':
            parts['lips'] = image[y-20:y+20, x-40:x+40]  # Assuming mouth width is around 80 pixels
    return parts

def preprocess_image(image_path):
    detector = MTCNN()
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    faces = detector.detect_faces(image_rgb)
    if not faces:
        raise ValueError("No faces detected in the image")
    
    keypoints = faces[0]['keypoints']
    parts = extract_face_parts(image_rgb, keypoints)
    
    required_parts = ['left_eye', 'right_eye', 'nose', 'lips']
    data = np.zeros((64, 64, 4), dtype=np.uint8)
    
    for idx, part in enumerate(required_parts):
        if part in parts and parts[part].shape[0] > 0 and parts[part].shape[1] > 0:
            img_part_gray = cv2.cvtColor(cv2.resize(parts[part], (64, 64)), cv2.COLOR_RGB2GRAY)
            data[..., idx] = img_part_gray
        else:
            raise ValueError(f"Failed to extract {part} from the image")
    
    data = np.expand_dims(data, axis=0)  # Add batch dimension
    return data

def predict_autism(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    return prediction[0][0] > 0.5  # Return True if the prediction is above 0.5, indicating autism

def predict_directory(directory_path, model_path):
    model = load_model(model_path)
    autistic_count = 0
    non_autistic_count = 0
    correct_predictions = 0
    total_predictions = 0
    
    for image_name in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_name)
        if os.path.isfile(image_path):
            try:
                is_autistic = predict_autism(image_path, model)
                total_predictions += 1
                
                # Determine ground truth from directory structure or filename
                true_label = 'autistic' in directory_path.lower() or 'autistic' in image_name.lower()
                
                if is_autistic == true_label:
                    correct_predictions += 1
                
                if is_autistic:
                    autistic_count += 1
                    print(f"The image {image_name} is predicted to be autistic.")
                else:
                    non_autistic_count += 1
                    print(f"The image {image_name} is predicted to be non-autistic.")
            except ValueError as e:
                print(f"Error processing {image_name}: {e}")

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Total autistic images: {autistic_count}")
    print(f"Total non-autistic images: {non_autistic_count}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Path to your trained model
model_path = r'Model\autism_detection_model.h5'

# Path to the directory containing images you want to predict
directory_path = r'test\autistic'

predict_directory(directory_path, model_path)
