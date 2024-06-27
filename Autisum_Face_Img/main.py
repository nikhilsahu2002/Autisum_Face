import cv2
import os
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Function to extract face parts
def extract_face_parts(image, keypoints):
    parts = {}
    h, w, _ = image.shape
    
    # Extract each part with a defined margin
    parts['left_eye'] = image[max(0, keypoints['left_eye'][1]-30):keypoints['left_eye'][1]+30,
                              max(0, keypoints['left_eye'][0]-30):keypoints['left_eye'][0]+30]
    parts['right_eye'] = image[max(0, keypoints['right_eye'][1]-30):keypoints['right_eye'][1]+30,
                               max(0, keypoints['right_eye'][0]-30):keypoints['right_eye'][0]+30]
    parts['nose'] = image[max(0, keypoints['nose'][1]-30):keypoints['nose'][1]+30,
                          max(0, keypoints['nose'][0]-30):keypoints['nose'][0]+30]
    parts['lips'] = image[max(0, keypoints['mouth_left'][1]-20):keypoints['mouth_left'][1]+20,
                          max(0, min(keypoints['mouth_left'][0], keypoints['mouth_right'][0])-40):
                          max(0, max(keypoints['mouth_left'][0], keypoints['mouth_right'][0])+40)]

    # Define additional parts like upper face, middle face, philtrum, and chin
    parts['upper_face'] = image[max(0, min(keypoints['left_eye'][1], keypoints['right_eye'][1]) - 50):
                                min(keypoints['left_eye'][1], keypoints['right_eye'][1])+30,
                                max(0, min(keypoints['left_eye'][0], keypoints['right_eye'][0]) - 50):
                                min(w, max(keypoints['left_eye'][0], keypoints['right_eye'][0]) + 50)]
    
    parts['middle_face'] = image[max(0, keypoints['nose'][1] - 40):keypoints['nose'][1]+40,
                                 max(0, keypoints['nose'][0] - 50):min(w, keypoints['nose'][0] + 50)]
    
    parts['philtrum'] = image[max(0, keypoints['nose'][1] - 20):keypoints['nose'][1]+20,
                              max(0, keypoints['mouth_left'][0] - 40):min(w, keypoints['mouth_right'][0] + 40)]
    
    parts['chin'] = image[max(0, max(keypoints['mouth_left'][1], keypoints['mouth_right'][1]) + 20):
                          min(h, keypoints['jaw_left'][1] + 60),
                          max(0, keypoints['jaw_left'][0] - 40):
                          min(w, keypoints['jaw_right'][0] + 40)]
    
    return parts

# Function to load images and extract parts
def load_images(class_path):
    parts_list = ['left_eye', 'right_eye', 'nose', 'lips', 'upper_face', 'middle_face', 'philtrum', 'chin']
    parts_data = {part: [] for part in parts_list}
    detector = MTCNN()
    image_counter = 0  # Counter to track number of processed images

    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}. Skipping...")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        if faces:
            keypoints = faces[0]['keypoints']
            keypoints['jaw_left'] = (keypoints['mouth_left'][0], keypoints['nose'][1] + 80)
            keypoints['jaw_right'] = (keypoints['mouth_right'][0], keypoints['nose'][1] + 80)
            parts = extract_face_parts(image_rgb, keypoints)
            for part, img_part in parts.items():
                if part in parts_list and isinstance(img_part, np.ndarray) and img_part.size > 0:
                    img_part_gray = cv2.cvtColor(cv2.resize(img_part, (64, 64)), cv2.COLOR_RGB2GRAY)
                    parts_data[part].append(img_part_gray)
                    print(f"{part.capitalize()} extracted and saved successfully for {os.path.basename(image_path)}.")
                else:
                    print(f"Failed to extract {part} for {os.path.basename(image_path)}.")

            image_counter += 1  # Increment image counter

    return parts_data

# Paths to image directories
data_dir = r'E:\Open CV\Autisum_Face_Img\train'
autism_class = r'autistic'
no_autism_class = r'non_autistic'

# Load and process images for both classes
autism_parts = load_images(os.path.join(data_dir, autism_class))
no_autism_parts = load_images(os.path.join(data_dir, no_autism_class))

# Function to concatenate face parts into a single array for training
def concatenate_parts(parts):
    num_samples = min(len(parts['left_eye']), len(parts['right_eye']), len(parts['nose']), len(parts['lips']),
                      len(parts['upper_face']), len(parts['middle_face']), len(parts['philtrum']), len(parts['chin']))
    
    data = np.zeros((num_samples, 64, 64, 8), dtype=np.uint8)
    
    data[..., 0] = np.array(parts['left_eye'][:num_samples])
    data[..., 1] = np.array(parts['right_eye'][:num_samples])
    data[..., 2] = np.array(parts['nose'][:num_samples])
    data[..., 3] = np.array(parts['lips'][:num_samples])
    data[..., 4] = np.array(parts['upper_face'][:num_samples])
    data[..., 5] = np.array(parts['middle_face'][:num_samples])
    data[..., 6] = np.array(parts['philtrum'][:num_samples])
    data[..., 7] = np.array(parts['chin'][:num_samples])
    
    return data

# Prepare training and test datasets
X_autism = concatenate_parts(autism_parts)
X_no_autism = concatenate_parts(no_autism_parts)
y_autism = np.ones(len(X_autism))
y_no_autism = np.zeros(len(X_no_autism))

# Combine data from both classes
X_data = np.concatenate((X_autism, X_no_autism))
y_data = np.concatenate((y_autism, y_no_autism))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42)

# Define the custom CNN model
def create_model():
    # Input layers
    image_input = Input(shape=(64, 64, 8))

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Adding dropout for regularization
    output = Dense(1, activation='sigmoid')(x)

    # Create model
    model = Model(inputs=image_input, outputs=output)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Create the model
model = create_model()

# Print model summary
model.summary()

# Data augmentation for training images
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define callbacks
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
]

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=100, callbacks=callbacks
)

# Save the trained model
model.save("New_Face_Model.h5")

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
