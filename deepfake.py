import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Step 1: Define Paths to Dataset
base_dir = 'Dataset'  # Replace with the actual path to your Dataset folder

train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')

# Subfolder paths
train_real_dir = os.path.join(train_dir, 'Real')
train_fake_dir = os.path.join(train_dir, 'Fake')
val_real_dir = os.path.join(validation_dir, 'Real')
val_fake_dir = os.path.join(validation_dir, 'Fake')
test_real_dir = os.path.join(test_dir, 'Real')
test_fake_dir = os.path.join(test_dir, 'Fake')

# Verify the number of images
print(f"Train Real: {len(os.listdir(train_real_dir))} images")
print(f"Train Fake: {len(os.listdir(train_fake_dir))} images")
print(f"Validation Real: {len(os.listdir(val_real_dir))} images")
print(f"Validation Fake: {len(os.listdir(val_fake_dir))} images")
print(f"Test Real: {len(os.listdir(test_real_dir))} images")
print(f"Test Fake: {len(os.listdir(test_fake_dir))} images")

# Step 2: Data Preprocessing and Augmentation
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Adjust if RAM is insufficient

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation and test data generator (no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Step 3: Build the Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Step 4: Train the Model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'deepfake_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10,  # Start with 10 epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping]
)

# Step 5: Evaluate the Model
best_model = tf.keras.models.load_model('deepfake_model_best.h5')
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 6: Visualize Results
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_plots.png')  # Save the plot
plt.show()

# Step 7: Save the Final Model
model.save('deepfake_detection_model_final.h5')
print("Model training completed and saved as 'deepfake_detection_model_final.h5'")