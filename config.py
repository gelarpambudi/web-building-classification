from flask import Flask
import os

IMG_FOLDER = './static/streetview/'
MODEL_PATH = './model/VGG16-0.3368-0.8985.h5'

app = Flask(__name__)
app.config["IMG_FOLDER"] = IMG_FOLDER
app.config["API_KEY"] = os.environ['GOOGLE_STREETVIEW_KEY']
app.config["MODEL_PATH"] = MODEL_PATH