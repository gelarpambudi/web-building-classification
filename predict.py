import requests
import json
import random
import os
import numpy as np
from PIL import Image
from tensorflow import keras

BASE_URL = "https://maps.googleapis.com"
METADATA_ENDPOINT = "/maps/api/metadata"
STREETVIEW_ENDPOINT = "/maps/api/streetview"

def is_metadata_exist(query_params):
    requests_url = BASE_URL + METADATA_ENDPOINT
    response = requests.get(url=requests_url, params=query_params).json()
    status = json.dumps(response["status"])
    if status == "OK":
        return True
    else:
        return False


def get_image(query_params, save_dir):
    requests_url = BASE_URL + METADATA_ENDPOINT
    response = requests.get(url=requests_url, params=query_params)
    image_name = "image_{}".format(random.randint(1,1000000))
    save_path = os.path.join(save_dir, image_name)
    with open(save_path, 'wb') as f:
        f.write(response.content)

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
    image_arr = image_arr.resize((224,224))
    image_arr = image_arr[:, :, :3]
    image_arr = np.expand_dims(image_arr, axis=0)

    preds_prob = model.predict_proba(image_arr, batch_size=10, verbose=1)
    highest_prob = np.amax(preds_prob[0])
    class_id = np.where(preds_prob == highest_prob)[1][0]
    inv_map = {v: k for k, v in img_classes.items()}  
    label = inv_map[class_id]

    return class_id, label, highest_prob


