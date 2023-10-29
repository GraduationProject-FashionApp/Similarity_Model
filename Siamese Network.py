import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = np.random.randint(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(2, activation='sigmoid')(x)
    return Model(input, x)

# Load data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

input_shape = train_images.shape[1:]

# Create pairs for training
digit_indices = [np.where(train_labels == i)[0] for i in range(10)]
train_pairs, train_labels = create_pairs(train_images, digit_indices)

train_labels = train_labels.astype(np.float32)

base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
processed_a = base_network(input_a)
processed_b = base_network(input_b)
distance = Lambda(euclidean_distance)([processed_a, processed_b])
model = Model([input_a, input_b], distance)

# Compile the model
rms = RMSprop()
model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms, metrics=['accuracy'])

# Train the model
model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels, batch_size=128, epochs=10)

def load_images_from_directory(directory_path, target_size=(28, 28)):
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory_path, filename)
            img = load_img(img_path, target_size=target_size, color_mode='grayscale')
            img_array = img_to_array(img).astype('float32') / 255.0
            img_array = img_array[..., np.newaxis]  # add a channel dimension
            images.append(img_array)
    return np.array(images)

# 경로 설정
path_to_first_folder = 'C:\\pants\\denim'
path_to_second_folder = 'C:\\pants\\training'

# 이미지 불러오기
first_folder_images = load_images_from_directory(path_to_first_folder)
second_folder_images = load_images_from_directory(path_to_second_folder)

# 평균 유사도 계산
similarities = []

for img_a in first_folder_images:
    for img_b in second_folder_images:
        prediction = model.predict([np.array([img_a]), np.array([img_b])])[0]
        similarities.append(prediction)

average_similarity = np.mean(similarities)

print(f"Average similarity between the two folders is: {average_similarity}")