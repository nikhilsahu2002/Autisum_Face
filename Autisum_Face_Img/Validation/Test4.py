import cv2
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
            parts['lips'] = image[y-20:y+20, x-40:x+40]

    # Additional features
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    
    # Upper face (forehead and eyes)
    upper_y = min(left_eye[1], right_eye[1]) - 50
    upper_x1 = min(left_eye[0], right_eye[0]) - 50
    upper_x2 = max(left_eye[0], right_eye[0]) + 50
    parts['upper_face'] = image[upper_y:min(left_eye[1], right_eye[1])+30, upper_x1:upper_x2]

    # Middle face (nose and cheeks)
    middle_y = nose[1] - 40
    middle_x1 = nose[0] - 50
    middle_x2 = nose[0] + 50
    parts['middle_face'] = image[middle_y:nose[1]+40, middle_x1:middle_x2]

    # Philtrum (groove below the nose, above the top lip)
    philtrum_y = nose[1]
    philtrum_x1 = mouth_left[0]
    philtrum_x2 = mouth_right[0]
    parts['philtrum'] = image[philtrum_y-20:philtrum_y+20, philtrum_x1:philtrum_x2]
    
    return parts

def is_autistic(image_path, model):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)

    if faces:
        keypoints = faces[0]['keypoints']
        parts = extract_face_parts(image_rgb, keypoints)
        
        num_samples = min(len(parts['left_eye']), len(parts['right_eye']), len(parts['nose']), 
                          len(parts['lips']), len(parts['upper_face']), len(parts['middle_face']), 
                          len(parts['philtrum']))

        data = np.zeros((num_samples, 64, 64, 7), dtype=np.uint8)
        data[..., 0] = np.array([cv2.resize(img, (64, 64)) for img in parts['left_eye'][:num_samples]])
        data[..., 1] = np.array([cv2.resize(img, (64, 64)) for img in parts['right_eye'][:num_samples]])
        data[..., 2] = np.array([cv2.resize(img, (64, 64)) for img in parts['nose'][:num_samples]])
        data[..., 3] = np.array([cv2.resize(img, (64, 64)) for img in parts['lips'][:num_samples]])
        data[..., 4] = np.array([cv2.resize(img, (64, 64)) for img in parts['upper_face'][:num_samples]])
        data[..., 5] = np.array([cv2.resize(img, (64, 64)) for img in parts['middle_face'][:num_samples]])
        data[..., 6] = np.array([cv2.resize(img, (64, 64)) for img in parts['philtrum'][:num_samples]])

        prediction = model.predict(data)
        return np.mean(prediction) >= 0.5
    else:
        return False

# Load the trained model
model_path = 'Face_Model.h5'
model = load_model(model_path)

# Provide the path to the image you want to test
image_path = r'Autisum_Face_Img\test\non_autistic\002.jpg'

# Test the model
if is_autistic(image_path, model):
    print(f"The person in {image_path} is predicted to be autistic.")
else:
    print(f"The person in {image_path} is predicted not to be autistic.")
