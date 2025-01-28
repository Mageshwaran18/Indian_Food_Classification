from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Data Preprocessing Function
def preprocess_data(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Resize Image Function
def resize_image(image, label):
    image = tf.image.resize(image, [224, 224])
    return image, label

# Data Augmentation Function
def augment_data(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_rotation(image, 0.2)
    image = tf.image.random_zoom(image, 0.2)
    return image, label

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(6, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
    )
    return model

# Get all image paths and labels
image_paths = []
labels = []
data_dir = "path/to/food/dataset"
class_names = os.listdir(data_dir)

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, img_name))
        labels.append(class_idx)

# Convert to numpy arrays
image_paths = np.array(image_paths)
labels = np.array(labels)

# Initialize Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Training loop for each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
    print(f"Training Fold {fold + 1}/{n_splits}")
    
    # Get paths for this fold
    train_paths = image_paths[train_idx]
    val_paths = image_paths[val_idx]
    
    # Create temporary directories for this fold
    train_fold_dir = "temp/train"
    val_fold_dir = "temp/val"
    
    # Create symbolic links for images in temp directories
    for path in train_paths:
        class_name = path.split(os.sep)[-2]
        os.makedirs(os.path.join(train_fold_dir, class_name), exist_ok=True)
        os.symlink(path, os.path.join(train_fold_dir, class_name, os.path.basename(path)))
    
    for path in val_paths:
        class_name = path.split(os.sep)[-2]
        os.makedirs(os.path.join(val_fold_dir, class_name), exist_ok=True)
        os.symlink(path, os.path.join(val_fold_dir, class_name, os.path.basename(path)))
    
    # Create datasets using image_dataset_from_directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_fold_dir,
        validation_split=None,
        batch_size=32,
        image_size=(224, 224),
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_fold_dir,
        validation_split=None,
        batch_size=32,
        image_size=(224, 224),
        shuffle=False
    )
    
    # Apply preprocessing and augmentation
    train_ds = train_ds.map(preprocess_data)
    train_ds = train_ds.map(augment_data)
    train_ds = train_ds.map(resize_image)
    val_ds = val_ds.map(resize_image)
    val_ds = val_ds.map(preprocess_data)

    # Configure dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    
    # Calculate class weights
    class_labels = np.concatenate([y for x, y in train_ds], axis=0)
    unique_classes = np.unique(class_labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=class_labels
    )
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    # Train the model
    model = create_model()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
        ]
    )
    
    # Clean up temp directories
    import shutil
    shutil.rmtree("temp")
