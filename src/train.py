# train.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from data_loader import load_data, load_test_data
from model import create_model

# Define paths and categories
data_dir = '../Data/train'
test_data_dir = '../Data/test'
categories = sorted(os.listdir(data_dir))

# Load and preprocess training data
data, labels = load_data(data_dir, categories)
data = data / 255.0
labels = to_categorical(labels, num_classes=len(categories))

# Load and preprocess test data
test_data = load_test_data(test_data_dir)
test_data = test_data / 255.0

# Split training data
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Explorative Data Analysis
# Distribution of classes
label_counts = np.sum(labels, axis=0)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=categories, y=label_counts)
plt.title('Distribution of Classes')
plt.xlabel('Category', labelpad=50)
plt.ylabel('Count')

# Adjust x-axis labels
xticks = ax.get_xticks()
xlabels = ax.get_xticklabels()
for i, label in enumerate(xlabels):
    if i % 2 == 0:
        ax.text(xticks[i], -0.05, label.get_text(), ha='center', rotation=0, transform=ax.get_xaxis_transform())
    else:
        ax.text(xticks[i], -0.08, label.get_text(), ha='center', rotation=0, transform=ax.get_xaxis_transform())
ax.set_xticklabels([])

plt.show()

# Display sample images from each category
def plot_sample_images(data, labels, categories, num_samples=5):
    plt.figure(figsize=(15, 15))
    for category_idx, category in enumerate(categories):
        category_indices = np.where(np.argmax(labels, axis=1) == category_idx)[0]
        selected_indices = np.random.choice(category_indices, num_samples, replace=False)
        for i, idx in enumerate(selected_indices):
            plt.subplot(len(categories), num_samples, category_idx * num_samples + i + 1)
            plt.imshow(data[idx])
            plt.axis('off')
            if i == 0:
                plt.ylabel(category)
    plt.show()

plot_sample_images(X_train, y_train, categories)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(X_train)

# Create model
model = create_model(input_shape=(100, 100, 3), num_classes=len(categories))

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=3), epochs=5, validation_data=(X_val, y_val))

# Evaluate the model on validation data
y_val_pred_prob = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_prob, axis=1)
y_val_true = np.argmax(y_val, axis=1)

# Compute confusion matrix and other metrics on validation date
conf_matrix = confusion_matrix(y_val_true, y_val_pred)
acc = accuracy_score(y_val_true, y_val_pred)
prec = precision_score(y_val_true, y_val_pred, average='weighted')
test_prec = precision_score(y_val_true, y_val_pred, average='weighted')
rec = recall_score(y_val_true, y_val_pred, average='weighted')
f1 = f1_score(y_val_true, y_val_pred, average='weighted')

# Evaluate the model on test data
y_test_pred_prob = model.predict(test_data)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Print the results
print(f'Accuracy: {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall: {rec:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model.save('saved_models/fruit_classifier_model.h5')