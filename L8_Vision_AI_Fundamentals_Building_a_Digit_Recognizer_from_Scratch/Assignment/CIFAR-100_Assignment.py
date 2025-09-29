# --- CIFAR-100 Assignment ---

import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 1. Dataset Setup
(cifar100_train_images, cifar100_train_labels), (cifar100_test_images, cifar100_test_labels) = keras.datasets.cifar100.load_data()
cifar100_train_images = cifar100_train_images.astype('float32') / 255.0
cifar100_test_images = cifar100_test_images.astype('float32') / 255.0
cifar100_train_labels_one_hot = keras.utils.to_categorical(cifar100_train_labels, num_classes=100)
cifar100_test_labels_one_hot = keras.utils.to_categorical(cifar100_test_labels, num_classes=100)
print("Training images shape:", cifar100_train_images.shape)
print("Testing images shape:", cifar100_test_images.shape)
print("Training labels shape:", cifar100_train_labels_one_hot.shape)
print("Testing labels shape:", cifar100_test_labels_one_hot.shape)

# 2. Model Building

# ANN Model
ann_cifar_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(100, activation='softmax')
])
ann_cifar_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann_cifar_model.summary()

# Basic CNN Model
basic_cnn_cifar_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(100, activation='softmax')
])
basic_cnn_cifar_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
basic_cnn_cifar_model.summary()

# 3. Model Training

early_stopping_cifar = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint_ann_cifar = keras.callbacks.ModelCheckpoint(
    filepath='best_ann_cifar_model.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)
model_checkpoint_basic_cnn_cifar = keras.callbacks.ModelCheckpoint(
    filepath='best_basic_cnn_cifar_model.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)

print("Training ANN model on CIFAR-100...")
ann_cifar_history = ann_cifar_model.fit(
    cifar100_train_images, cifar100_train_labels_one_hot,
    epochs=30, batch_size=64,
    validation_data=(cifar100_test_images, cifar100_test_labels_one_hot),
    callbacks=[early_stopping_cifar, model_checkpoint_ann_cifar]
)
print("Training Basic CNN model on CIFAR-100...")
basic_cnn_cifar_history = basic_cnn_cifar_model.fit(
    cifar100_train_images, cifar100_train_labels_one_hot,
    epochs=30, batch_size=64,
    validation_data=(cifar100_test_images, cifar100_test_labels_one_hot),
    callbacks=[early_stopping_cifar, model_checkpoint_basic_cnn_cifar]
)

# 4. Model Evaluation

ann_cifar_model.load_weights('best_ann_cifar_model.weights.h5')
basic_cnn_cifar_model.load_weights('best_basic_cnn_cifar_model.weights.h5')
loss_ann_cifar, acc_ann_cifar = ann_cifar_model.evaluate(cifar100_test_images, cifar100_test_labels_one_hot, verbose=0)
loss_basic_cnn_cifar, acc_basic_cnn_cifar = basic_cnn_cifar_model.evaluate(cifar100_test_images, cifar100_test_labels_one_hot, verbose=0)
print(f"ANN CIFAR-100 Test Loss: {loss_ann_cifar:.4f}, Accuracy: {acc_ann_cifar:.4f}")
print(f"Basic CNN CIFAR-100 Test Loss: {loss_basic_cnn_cifar:.4f}, Accuracy: {acc_basic_cnn_cifar:.4f}")

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(ann_cifar_history.history['accuracy'], label='ANN Train')
plt.plot(ann_cifar_history.history['val_accuracy'], label='ANN Val')
plt.plot(basic_cnn_cifar_history.history['accuracy'], label='CNN Train')
plt.plot(basic_cnn_cifar_history.history['val_accuracy'], label='CNN Val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(ann_cifar_history.history['loss'], label='ANN Train')
plt.plot(ann_cifar_history.history['val_loss'], label='ANN Val')
plt.plot(basic_cnn_cifar_history.history['loss'], label='CNN Train')
plt.plot(basic_cnn_cifar_history.history['val_loss'], label='CNN Val')
plt.title('Loss')
plt.legend()
plt.suptitle('CIFAR-100 Model Training History')
plt.tight_layout()
plt.show()

# Confusion Matrix & Classification Report for best model
best_model = basic_cnn_cifar_model if acc_basic_cnn_cifar > acc_ann_cifar else ann_cifar_model
predictions = best_model.predict(cifar100_test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(cifar100_test_labels_one_hot, axis=1)
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(true_labels, predicted_labels))

# 5. Prediction Analysis: Show some correct/incorrect predictions
correct_indices = np.where(predicted_labels == true_labels)[0]
incorrect_indices = np.where(predicted_labels != true_labels)[0]
num_to_show = min(20, len(correct_indices), len(incorrect_indices))
plt.figure(figsize=(10, 4))
for i in range(num_to_show):
    plt.subplot(2, num_to_show, i+1)
    plt.imshow(cifar100_test_images[correct_indices[i]])
    plt.title(f"Pred: {predicted_labels[correct_indices[i]]}", color='green', fontsize=8)
    plt.axis('off')
    plt.subplot(2, num_to_show, num_to_show+i+1)
    plt.imshow(cifar100_test_images[incorrect_indices[i]])
    plt.title(f"P:{predicted_labels[incorrect_indices[i]]}\nT:{true_labels[incorrect_indices[i]]}", color='red', fontsize=8)
    plt.axis('off')
plt.suptitle("CIFAR-100: Correct (Top) & Incorrect (Bottom) Predictions")
plt.tight_layout()
plt.show()
# --- End CIFAR-100 Assignment ---