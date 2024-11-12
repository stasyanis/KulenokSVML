from flask import Flask, request, render_template, redirect, url_for
import os
from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image(image_path):
    # Открываем изображение
    img = Image.open(image_path)
    # Изменяем размер изображения
    img = img.resize((224, 224))  # Замените на нужный размер
    # Преобразуем в массив NumPy
    img_array = np.array(img)
    # Нормализуем значения пикселей
    img_array = img_array / 255.0  # Преобразуем в диапазон [0, 1]
    # Добавляем размерность для батча
    img_array = np.expand_dims(img_array, axis=0)
    return img_array