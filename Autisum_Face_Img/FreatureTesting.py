import cv2
import os
import numpy as np
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

def load_images(class_path):
    parts_list = ['left_eye', 'right_eye', 'nose', 'lips', 'upper_face', 'middle_face', 'philtrum']
    parts_data = {part: [] for part in parts_list}
    detector = MTCNN()

    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        if faces:
            keypoints = faces[0]['keypoints']
            parts = extract_face_parts(image_rgb, keypoints)
            for part, img_part in parts.items():
                if part in parts_list and img_part.shape[0] > 0 and img_part.shape[1] > 0:
                    img_part_gray = cv2.cvtColor(cv2.resize(img_part, (64, 64)), cv2.COLOR_RGB2GRAY)
                    parts_data[part].append(img_part_gray)
                    print(f"{part.capitalize()} extracted and saved successfully for {os.path.basename(image_path)}.")
                else:
                    print(f"Failed to extract {part} for {os.path.basename(image_path)}.")
    return parts_data

data_dir = r'E:\Open CV\Autisum_Face_Img\train'
autism_class = r'autistic'
no_autism_class = r'non_autistic'

autism_parts = load_images(os.path.join(data_dir, autism_class))
no_autism_parts = load_images(os.path.join(data_dir, no_autism_class))

def concatenate_parts(parts):
    num_samples = min(len(parts['left_eye']), len(parts['right_eye']), len(parts['nose']), len(parts['lips']),
                      len(parts['upper_face']), len(parts['middle_face']), len(parts['philtrum']))
    data = np.zeros((num_samples, 64, 64, 7), dtype=np.uint8)
    
    data[..., 0] = np.array(parts['left_eye'][:num_samples])
    data[..., 1] = np.array(parts['right_eye'][:num_samples])
    data[..., 2] = np.array(parts['nose'][:num_samples])
    data[..., 3] = np.array(parts['lips'][:num_samples])
    data[..., 4] = np.array(parts['upper_face'][:num_samples])
    data[..., 5] = np.array(parts['middle_face'][:num_samples])
    data[..., 6] = np.array(parts['philtrum'][:num_samples])
    
    return data

X_autism = concatenate_parts(autism_parts)
X_no_autism = concatenate_parts(no_autism_parts)
y_autism = np.ones(len(X_autism))
y_no_autism = np.zeros(len(X_no_autism))

X_train = np.concatenate((X_autism, X_no_autism))
y_train = np.concatenate((y_autism, y_no_autism))

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the custom CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 7)),  # Change input shape to 7 channels
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

batch_sizes = [32]

for batch_size in batch_sizes:
    print(f"\nTraining with batch size: {batch_size}")
    
    # Create the model
    model = create_model()
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    datagen.fit(X_train)
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    # Train the model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=max(1, len(X_train) // 32),
        epochs=5,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Save the model
    model.save(f"autism_detection_model_batch_{batch_size}.h5")
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Batch Size: {batch_size}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
