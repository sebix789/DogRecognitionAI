import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Input paths
model_directory = '/kaggle/input/dogs-faceid-model/siamese_recognition_model.keras'
data_directory = '/kaggle/input/habana-dataset/habana/'
output_directory = '/kaggle/working/dog_faceid_model.keras'


# Load the model
def load_model():
    print("Load existing model...")
    model_path = os.path.join(model_directory)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model


# Create Embeddings
def create_embedding_model(pretrained_model):
    if pretrained_model is None:
        print("Pretrained model is None. Check the model loading.")
        return None
    
    try:
        inputs = pretrained_model.inputs
        outputs = pretrained_model.layers[-2].output
        print("Model input and output successfully retrieved.")
        
    except Exception as e:
        print(f"Error while getting model input/output: {e}")
        return None

    base_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    embeddings = base_model(inputs)
    normalized_embeddings = tf.keras.layers.LayerNormalization(axis=1)(embeddings)
    embedding_model = tf.keras.Model(inputs, normalized_embeddings)

    return embedding_model


# Build Siamese Network
def build_siamese_network(embedding_model):
    # Inputs for image pairs
    input_a = tf.keras.layers.Input(shape=(224, 224, 3))
    input_b = tf.keras.layers.Input(shape=(224, 224, 3))
    
    # Generate the embeddings for the image pairs
    embedding_a = embedding_model(input_a)
    embedding_b = embedding_model(input_b)
    
    # Calculate disstance between embeddings
    distance = tf.keras.layers.Subtract()([embedding_a, embedding_b])
    normalized_distance = tf.keras.layers.LayerNormalization(axis=1)(distance)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(normalized_distance)
    
    siamese_model = tf.keras.Model([input_a, input_b], outputs)
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return siamese_model


# Preprocess Image
def preprocess_image(image, label, is_training=True):
    image = tf.image.resize(image, (224, 224))
    
    # Data Augmentation
    if is_training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.95, upper=1.05)
        image = tf.image.random_saturation(image, lower=0.95, upper=1.05)
        
        # Random Zoom Simulation
        zoom_factor = tf.random.uniform([], minval=0.9, maxval=1.1)
        new_size = tf.cast(tf.shape(image)[0:2], tf.float32) * zoom_factor
        image = tf.image.resize(image, tf.cast(new_size, tf.int32))
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        
        # Rotation
        rotations = random.choice([0, 1])
        image = tf.image.rot90(image, k=rotations)
        
    image = image / 255.0
    return image, label


# Load Custom Dataset
def load_custom_images():
    images = []
    labels = []
    custom_label = 120
    
    for image_file in os.listdir(data_directory):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            img_path = os.path.join(data_directory, image_file)
            img = tf.keras.preprocessing.image.load_img(img_path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            preprocessed_image, _ = preprocess_image(img_array, custom_label, is_training=True)
            
            images.append(preprocessed_image)
            labels.append(custom_label)
    
    return np.array(images), np.array(labels)


# Load Stanford Dogs Dataset
def load_stanford_dogs(max_samples=None):
    data, info = tfds.load('stanford_dogs', with_info=True, as_supervised=True)
    train_data = data['train'].map(lambda img, lbl: preprocess_image(img, lbl, is_training=True))
    test_data = data['test'].map(lambda img, lbl: preprocess_image(img, lbl, is_training=False))
    
    if max_samples is not None:
        train_data = train_data.shuffle(buffer_size=10000).take(max_samples)
        test_data = test_data.shuffle(buffer_size=10000).take(max_samples)
    
    return train_data, test_data


# Combine datasets
def combine_datasets(stanford_images, stanford_labels, custom_images, custom_labels):
    combined_images = np.concatenate([stanford_images, custom_images], axis=0)
    combined_labels = np.concatenate([stanford_labels, custom_labels], axis=0)
    
    return combined_images, combined_labels


# Create pairs to compare
def create_pairs(images, labels, num_classes):
    pairs = []
    pair_labels = []
    
    class_images = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for idx1 in range(len(images)):
        current_class = labels[idx1]
        
        idx2 = random.choice(class_images[current_class])
        pairs.append([images[idx1], images[idx2]])
        pair_labels.append(1)
        
        negative_class = random.choice([x for x in range(num_classes) if x != current_class])
        idx2 = random.choice(class_images[negative_class])
        pairs.append([images[idx1], images[idx2]])
        pair_labels.append(0)
    
    return np.array(pairs), np.array(pair_labels)    


# Train Model
def train_model(model, pairs, pair_labels):
    images_a = pairs[:, 0]
    images_b = pairs[:, 1]
    
    history = model.fit([images_a, images_b], pair_labels, batch_size=32, epochs=30, validation_split=0.2)
    
    model.save(output_directory)
    print(f"Model saved to {output_directory}")
    
    return history


# Visualize the metrics
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy over epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss over epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    

# Evaluate the model
def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    
    predictions = model.predict(test_data)
    predicted_labels = (predictions > 0.5).astype("int32")
    
    # Classification Report - includes precision, recall and f1-score
    print("\nClassification Report")
    print(classification_report(test_labels, predicted_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, predicted_labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Different", "Same"], yticklabels=["Different", "Same"])
    plt.title("Confusion Matrix")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.show()
    
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    


def load_model():
    print("Load existing model...") 
    model_path = os.path.join(os.path.dirname(__file__), 'dog_faceid_model.keras')
    print(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    
    model = tf.keras.models.load_model(model_path)
    return model
    
    
if __name__ == '__main__':
    print("Load stanford dogs dataset...")
    stanford_train, _ = load_stanford_dogs(max_samples=1000) 
    print("Load custom dogs dataset...")
    custom_images, custom_labels = load_custom_images()

    stanford_images = []
    stanford_labels = []
    
    for img, lbl in stanford_train:
        stanford_images.append(img.numpy())
        stanford_labels.append(lbl.numpy())

    stanford_images = np.array(stanford_images)
    stanford_labels = np.array(stanford_labels)

    combined_images, combined_labels = combine_datasets(stanford_images, stanford_labels, custom_images, custom_labels)

    # Create pairs
    print("\nCreating pairs...")
    pairs, pair_labels = create_pairs(combined_images, combined_labels, num_classes=121)

    if len(pairs) == 0 or len(pair_labels) == 0:
        print("No pairs created for training. Check the datasets.")
    else:
        model = load_model()
        embedding_model = create_embedding_model(model)
        siamese_model = build_siamese_network(embedding_model)
        
        history = train_model(siamese_model, pairs, pair_labels)
        plot_history(history)
        
        evaluate_model(siamese_model, combined_images, combined_labels)