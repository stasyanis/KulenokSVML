import pickle
import math
from enum import auto, Enum
from werkzeug.utils import secure_filename
import os

import numpy as np
import keras
from keras import ops

import pandas as pd
from flask import Flask, render_template, url_for, request, jsonify
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score, \
    accuracy_score, precision_score, recall_score, f1_score

from model.use_neo import new_neuron
from model.neo import SingleNeuron

from flask import Flask, request, render_template, redirect, url_for
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "lin_reg"},
        {"name": "Лаба 2", "url": "log_reg"},
        {"name": "Лаба 3", "url": "knn"},
        {"name": "Лаба 4", "url": "dec_tree"},
        {"name": "Лаба 5", "url": "neouron"},
        {"name": "Лаба 6", "url": "pickneuron"}]

heights = ["нормален", "немного низкорослый", "карлик", "высокий"]

genders = ["женщина", "мужчина"]

wines = ["Выдающееся", "Качественное", "Местное"]

classes = ["футболка", "брюки", "кофта", "платье", "пальто", "сандали", "рубашка", "кроссовок", "сумка", "ботинок"]

UPLOAD_FOLDER = 'uploads'
app.config['uploads'] = UPLOAD_FOLDER
sequ_model = tf.keras.models.load_model('model/Sequential_model.keras')

loaded_model_linear_regression = pickle.load(open('model/legs_model_l1', 'rb'))
loaded_model_logistic_regression = pickle.load(open('model/baby_pickle_file_l2', 'rb'))
loaded_model_knn = pickle.load(open('model/model_genbylegs_l3', 'rb'))
loaded_model_decision_tree = pickle.load(open('model/wine_model_l4', 'rb'))
load_clothes = keras.models.load_model("model/classification_model.h5")


def preproces_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = 1 - img_array
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Куленком Станиславом Владимировичем",
                           menu=menu)


@app.route("/lin_reg", methods=['POST', 'GET'])
def f_lab1():

    if request.method == 'GET':
        return render_template('lab1.html', title="Метод Линейной регрессии", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred = loaded_model_linear_regression.predict(X_new)
        return render_template('lab1.html', title="Метод Линейной регрессии", menu=menu,
                               class_model=f"Размер ноги: {math.ceil(pred[0])}")


@app.route("/log_reg", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Метод Логистической регрессии", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred = loaded_model_logistic_regression.predict(X_new)

        return render_template('lab2.html', title="Метод Логистической регрессии", menu=menu,
                               class_model=f"Ваш ребёнок {heights[pred[0]]}")


@app.route("/knn", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab3.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model=f"Это: {genders[pred[0]]}")



@app.route("/dec_tree", methods=['POST', 'GET'])
def f_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Метод decision_tree", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7']),
                           float(request.form['list8']),
                           float(request.form['list9']),
                           float(request.form['list10']),
                           float(request.form['list11']),
                           float(request.form['list12']),
                           int(request.form['list13'])]])
        pred = loaded_model_decision_tree.predict(X_new)
        return render_template('lab4.html', title="Метод дерева решений", menu=menu,
                               class_model=f"Категория вина: {wines[pred[0]]}")

@app.route("/neouron", methods=['POST', 'GET'])
def f_lab5():
    if request.method == 'GET':
        return render_template('lab5.html', title="обучение нейрона", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7']),
                           float(request.form['list8'])]])
        predictions = new_neuron.forward(X_new)

        return render_template('lab5.html', title="Первый нейрон диабет", menu=menu,
                               class_model="Ваш диагноз: " + str(*np.where(predictions >= 0.5, 'Диабета нет', 'Диабет есть')))

@app.route('/pickneuron', methods=['GET', 'POST'])
def lab6():
    if request.method == 'GET':
        return render_template('lab6.html', title="Cвёрт сеть", menu=menu, class_model='')
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        file_path = os.path.join(app.config['uploads'], file.filename)
        file.save(file_path)
        img_array = preproces_image(file_path)
        predictions = sequ_model.predict(img_array)
        prediction = np.argmax(predictions, axis=1)


        return render_template('lab6.html', title="Свёрт сеть", menu=menu,
                               class_model=f"Одёжа:  {classes[prediction[0]]}")

@app.route("/api/dia", methods=['GET'])
def neuron_api():

    X_new = np.array([float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7']),
                           float(request.form['list8'])])
    y_pred_class = load_clothes.predict(X_new)

    return jsonify(msg=(y_pred_class >= 0.5, 'Диабет есть', 'Диабета нет'))



@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('sepal_length')),
                       float(request.args.get('sepal_width')),
                       float(request.args.get('petal_length')),
                       float(request.args.get('petal_width'))]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(sort=pred[0])


@app.route("/api/height", methods=['GET'])
def get_api():
    X_new = np.array([[float(request.args['list1']),
                       float(request.args['list2']),
                       float(request.args['list3'])]])
    pred = loaded_model_logistic_regression.predict(X_new)

    return jsonify(msg=(heights[pred[0]]))


@app.route("/api/gender", methods=['GET'])
def get_gender_api():
    X_new = np.array([[float(request.args['list1']),
                       float(request.args['list2']),
                       float(request.args['list3'])]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(msg=(genders[pred[0]]))


@app.route("/api/foot", methods=['GET'])
def get_foot_size_api():
    X_new = np.array([[float(request.args['list1']),
                       float(request.args['list2']),
                       float(request.args['list3'])]])
    pred = loaded_model_linear_regression.predict(X_new)

    return jsonify(msg=(math.ceil(pred[0])))

@app.route("/api/wine", methods=['GET'])
def get_wine_api():
    X_new = np.array([[float(request.args['list1']),
                       float(request.args['list2']),
                       float(request.args['list3']),
                       float(request.args['list4']),
                       float(request.args['list5']),
                       float(request.args['list6']),
                       float(request.args['list7']),
                       float(request.args['list8']),
                       float(request.args['list9']),
                       float(request.args['list10']),
                       float(request.args['list11']),
                       float(request.args['list12']),
                       int(request.args['list13'])]])
    pred = loaded_model_decision_tree.predict(X_new)

    return jsonify(msg=(wines[pred[0]]))


@app.route('/api_v2', methods=['get'])
def get_sort_v2():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['sepal_length']),
                       float(request_data['sepal_width']),
                       float(request_data['petal_length']),
                       float(request_data['petal_width'])]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(sort=pred[0])


if __name__ == "__main__":
    app.run(debug=True)
