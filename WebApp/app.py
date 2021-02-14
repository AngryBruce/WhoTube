import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from tensorflow import keras

assert tf.__version__ >= "2.0"

import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/model_in_use.h5'

#Load your own trained model
model = keras.models.load_model(MODEL_PATH)

print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((299, 299))

    img.save("./uploads/image.png")

    x = tf.keras.preprocessing.image.img_to_array(img)

    x = keras.applications.xception.preprocess_input(x,data_format =  "channels_last")

    x = tf.reshape(x, shape = (1,299,299,3))
    preds = model.predict(x,batch_size=1)

    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)


        class_names = ["ATHLEAN-X",
                       "AsapSCIENCE",
                       "AzzyLand",
                       "BRIGHT SIDE",
                       "Brave Wilderness",
                       "Brent Rivera",
                       "CaseyNeistat",
                       "Chad Wild Clay",
                       "Chloe Ting",
                       "CollegeHumor",
                       "Collins Key",
                       "CrashCourse",
                       "DanTDM",
                       "Dang Matt Smith",
                       "David Dobrik",
                       "Dude Perfect",
                       "FGTeeV",
                       "FaZe Rug",
                       "FitnessBlender",
                       "Good Mythical Morning",
                       "Guava Juice",
                       "Infinite",
                       "Jake Paul",
                       "James Charles",
                       "Jelly",
                       "JennaMarbles",
                       "LTT",
                       "LazarBeam",
                       "Lilly Singh",
                       "Liza Koshy",
                       "Logan Paul",
                       "Lucas and Marcus",
                       "Markiplier",
                       "Marques Brownlee",
                       "Matthew Santoro",
                       "Miranda Sings",
                       "MrBeast",
                       "MyLifeAsEva",
                       "Ninja",
                       "Pencilmation",
                       "PewDiePie",
                       "Preston",
                       "Reaction Time",
                       "Rosanna Pansino",
                       "SSSniperWolf",
                       "SmarterEveryDay",
                       "Smosh",
                       "Tasty",
                       "The Royalty Family",
                       "Troom Troom",
                       "Unbox Therapy",
                       "Zach Choi ASMR",
                       "jeffreestar",
                       "nigahiga"]



        proba_dict = [(prob, class_index, class_names[class_index]) for class_index, prob, in enumerate(preds[0])]
        decode_prediction_top = sorted(proba_dict, reverse = True, key = lambda class_prob: class_prob[0])[:3]  #top3
        probability_top, index_top, class_name_top = zip(*decode_prediction_top)



        result_name_1 = str(class_name_top[0])               # Convert to string
        result_name_2 = str(class_name_top[1])
        result_proba_1 = str(int(probability_top[0]*100))

        result_final = "{name_1} - {prob_1}%".format(name_1=result_name_1, prob_1 =result_proba_1 )
        return jsonify(result = result_final)

    return None


if __name__ == '__main__':

    # Serve the app with gevent
     http_server = WSGIServer(('127.0.0.1', 5000), app)
     http_server.serve_forever()
