import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# Load CIFAR-10 dataset and select a small subset (e.g., 500 images) to reduce memory usage
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
sample_size = 500  # Adjust this to control memory usage
x_test_sample = x_test[:sample_size]
y_test_sample = y_test[:sample_size]

# Resize images to 128x128 to further reduce memory usage
img_size = 128
x_test_resized = tf.image.resize(x_test_sample, (img_size, img_size))

# Convert labels to categorical format for 10 classes
y_test_categorical = to_categorical(y_test_sample, 10)

# Function to evaluate a model on a subset and compute inference time
def evaluate_model(model, x_test, y_test, preprocess_func):
    # Preprocess test images
    x_test_processed = preprocess_func(x_test)

    # Add global average pooling and dense layer for CIFAR-10 classification
    model_with_top = tf.keras.models.Sequential([
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model_with_top.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Evaluate model
    start_time = time.time()
    test_loss, test_accuracy = model_with_top.evaluate(x_test_processed, y_test, verbose=0)
    end_time = time.time()
    inference_time = end_time - start_time

    # Generate predictions and evaluation metrics
    y_pred = model_with_top.predict(x_test_processed)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    report = classification_report(y_true, y_pred_classes, digits=4)
    confusion = confusion_matrix(y_true, y_pred_classes)

    return test_accuracy, test_loss, inference_time, report, confusion

# Evaluate each model one at a time to keep memory usage low
models = {
    'InceptionV3': (InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3)), inception_preprocess),
    'ResNet50': (ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3)), resnet_preprocess),
    'VGG16': (VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3)), vgg_preprocess),
}

for model_name, (model, preprocess_func) in models.items():
    print(f"\nEvaluating model: {model_name}")
    model.trainable = False  # Freeze layers to use as a feature extractor

    # Evaluate the model
    test_accuracy, test_loss, inference_time, report, confusion = evaluate_model(model, x_test_resized, y_test_categorical, preprocess_func)

    # Print model analysis results
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", confusion)