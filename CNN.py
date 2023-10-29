import tensorflow as tf
import numpy as np
from PIL import Image
import os

def load_and_preprocess_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28, 1)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)

# 특징 추출기로 모델 변경
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# 두 폴더에서 모든 이미지 파일 목록 가져오기
folder_path1 = "/content/sample_data/denim"
folder_path2 = "/content/sample_data/top"
image_files1 = [os.path.join(folder_path1, f) for f in os.listdir(folder_path1) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files2 = [os.path.join(folder_path2, f) for f in os.listdir(folder_path2) if f.endswith(('.png', '.jpg', '.jpeg'))]

similarities = []

for img_file1 in image_files1:
    image1 = load_and_preprocess_image(img_file1)
    feature1 = feature_extractor.predict(image1)

    for img_file2 in image_files2:
        image2 = load_and_preprocess_image(img_file2)
        feature2 = feature_extractor.predict(image2)

        cos_sim = np.dot(feature1, feature2.T) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        similarities.append(cos_sim[0][0])

# 모든 유사도 값의 평균 계산
average_similarity = np.mean(similarities)
print(f"Average Cosine Similarity: {average_similarity}")