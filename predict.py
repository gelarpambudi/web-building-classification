import requests
import json
import random
import os
import numpy as np
from PIL import Image
from cv2 import resize
from tensorflow import keras

GOOGLE_BASE_URL = "https://maps.googleapis.com"
GOOGLE_METADATA_ENDPOINT = "/maps/api/streetview/metadata"
GOOGLE_STREETVIEW_ENDPOINT = "/maps/api/streetview"
OSC_BASE_URL = "https://api.openstreetcam.org"
OSC_STREETVIEW_ENDPOINT = "/2.0/photo/"


def is_image_exist_google(query_params):
    requests_url = GOOGLE_BASE_URL + GOOGLE_METADATA_ENDPOINT
    response = requests.get(url=requests_url, params=query_params).json()
    status = response["status"]
    if status == "OK":
        return True
    else:
        return False


def get_image_google(query_params, save_dir):
    requests_url = GOOGLE_BASE_URL + GOOGLE_STREETVIEW_ENDPOINT
    response = requests.get(url=requests_url, params=query_params)
    image_name = "image_{}.jpg".format(random.randint(1,1000000))
    save_path = os.path.join(save_dir, image_name)
    with open(save_path, 'wb') as f:
        f.write(response.content)
        
    return save_path


def get_image_osc(query_params, save_dir):
    request_url = OSC_BASE_URL + OSC_STREETVIEW_ENDPOINT
    response = requests.get(url=request_url, params=query_params)
    api_code = json.dumps(response["status"]["apiCode"])
    if api_code == "600":
        image_url = json.dumps(response["result"]["data"][0]["fileurlLTh"]).replace('"','')
        image_name = "image_{}.jpg".format(random.randint(1,1000000))
        image_req = requests.get(url=image_url)
        save_path = os.path.join(save_dir, image_name)
        with open(save_path, 'wb') as f:
            f.write(image_req.content)

        return save_path


def load_model(model_path):
    trained_model = keras.models.load_model(model_path)
    return trained_model


def predict_image(model, img_path):
    img_classes = {
        'apartment': 0,
        'church': 1,
        'garage': 2,
        'house': 3,
        'industrial': 4,
        'officebuilding': 5,
        'retail': 6,
        'roof': 7
    }

    img = Image.open(img_path)
    image_arr = np.array(img, dtype=np.float64)
    image_arr = resize(image_arr, (224,224))
    image_arr = image_arr[:, :, :3]
    image_arr = np.expand_dims(image_arr, axis=0)

    preds_prob = model.predict_proba(image_arr, batch_size=10, verbose=1)
    highest_prob = np.amax(preds_prob[0])
    class_id = np.where(preds_prob == highest_prob)[1][0]
    inv_map = {v: k for k, v in img_classes.items()}  
    label = inv_map[class_id]

    return class_id, label, highest_prob


