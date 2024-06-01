# Image Classification Project
This project focuses on building an image classification model to distinguish between "real" and "fake" images using deep learning techniques. The dataset is preprocessed, augmented, and then used to train a convolutional neural network (CNN) with a transfer learning approach using the VGG16 model.

## Table of Contents
Installation
Data Preparation
Data Augmentation
Model Training
Evaluation
Usage
Results
Installation
To run this project, you need to install the following libraries:


!pip install tensorflow opencv-python
Data Preparation
Loading and Resizing Images

Images are loaded from the dataset and resized to 224x224 pixels.

def load_and_resize_images(folder_path):
    # Implementation here
Example code for loading real and fake images:


real_folder_path = "/content/drive/MyDrive/DataSet/real_folder"
fake_folder_path = "/content/drive/MyDrive/DataSet/fake_folder"
real_images, real_labels = load_and_resize_images(real_folder_path)
fake_images, fake_labels = load_and_resize_images(fake_folder_path)
Combining and Shuffling Data


all_images = np.concatenate([real_images, fake_images], axis=0)
all_labels = np.concatenate([real_labels, fake_labels], axis=0)
Data Augmentation
To artificially increase the dataset size, data augmentation techniques such as rotation, width shift, height shift, shear, zoom, and horizontal flip are applied using TensorFlow's ImageDataGenerator.

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
Augmented images and labels are created and resized:


# Example code for generating augmented data
augmented_images_list = []
augmented_labels_list = []

for i in range(len(original_images)):
    # Implementation here
Model Training
A Convolutional Neural Network (CNN) is built using the VGG16 model pre-trained on ImageNet. The model is trained with early stopping to prevent overfitting.


model = Sequential()
# Add VGG16 base model
# Add custom layers
# Compile the model with Adam optimizer
Training the Model
python
Copy code
history = model.fit(datagen.flow(X_train_normalized, y_train, batch_size=32),
                    epochs=25,
                    validation_data=(X_test_normalized, y_test),
                    callbacks=[early_stopping])
Evaluation
Evaluate the model's performance on the test set and visualize the results.

accuracy = model.evaluate(X_test_normalized, y_test)[1]
print("Test Accuracy:", accuracy)
Plotting Learning Curves



plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.show()

ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
Usage
To use the trained model to predict new images:


new_image_paths = ["/content/real_image.png", "/content/fake_image.png"]
new_images = [preprocess_image(path) for path in new_image_paths]
predictions = model.predict(new_images)
binary_predictions = (predictions > 0.7).astype(int)
Results
Display results with the new images:


for i, (path, prediction) in enumerate(zip(new_image_paths, binary_predictions)):
    label = "Real" if prediction == 0 else "Fake"
    img = Image.open(path)
    plt.subplot(1, len(new_image_paths), i + 1)
    plt.imshow(img)
    plt.title(f"Prediction: {label}")
    plt.axis("off")
    print(f"Image {i + 1}: {label} - {path}")

plt.show()
