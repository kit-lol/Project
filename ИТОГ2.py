import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Загрузка обученной модели нейросети
model = tf.keras.models.load_model('mnist_model.h5')

# Путь к папке с изображениями
folder_path = 'цифры для проекта'

# Получение списка файлов из папки
image_files = os.listdir(folder_path)

# Выбор случайного изображения из списка
random_image_name = random.choice(image_files)
random_image_path = os.path.join(folder_path, random_image_name)

# Загрузка и отображение выбранного изображения
img = Image.open(random_image_path).convert('L')  # Преобразование изображения в черно-белый формат
plt.imshow(img, cmap='gray')
plt.show()

# Предобработка изображения для подачи в нейросеть
img = img.resize((28, 28))  # Изменение размера до 28x28 пикселей
img_array = np.array(img)  # Преобразование изображения в массив numpy
img_array = img_array / 255.0  # Нормализация значений пикселей
img_array = np.expand_dims(img_array, axis=0)  # Добавление измерения пакета (batch dimension)

# Предсказание метки для изображения с помощью обученной нейросети
predictions = model.predict(img_array)
predicted_label = np.argmax(predictions)
print('Цифра:', predicted_label)