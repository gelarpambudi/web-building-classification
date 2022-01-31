import tensorflow as tf
import os
from config import app
from flask import request, render_template, flash, redirect, session
from predict import is_image_exist_google, predict_image, get_image_google, get_image_osc, load_model

model = load_model(app.config["MODEL_PATH"])

@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        longitude = str(request.form['Longitude'])
        latitude = str(request.form['Latitude'])
        api_options = str(request.form['api_options'])

        if api_options == "google":
            API_KEY = app.config["API_KEY"]
            query_params = {
                "location": f"{latitude},{longitude}",
                "size": "800x800",
                "key": API_KEY,
                "return_error_code": "true",
                "source": "outdoor",
                "radius": 10
            }
            if is_image_exist_google(query_params) == True:
                save_dir = app.config['IMG_FOLDER']
                image_path = get_image_google(query_params, save_dir)
                class_id, label, prob = predict_image(model, image_path)
                return render_template(
                    "predict.html",
                    img_path=os.path.relpath(image_path, './static'),
                    class_id=class_id,
                    preds_label=label,
                    preds_prob=prob
                    )
            else:
                flash(u'Image does not exists')
                return redirect(request.url)
        elif api_options == "osc":
            query_params = {
                "lng": longitude,
                "lat": latitude,
                "projection": "PLANE",
                "projectionYaw": "0"
            }
            image_path = get_image_osc(query_params)
            if image_path is not None:
                class_id, label, prob = predict_image(model, image_path)
                return render_template(
                    "predict.html",
                    img_path=os.path.relpath(image_path, './static'),
                    class_id=class_id,
                    preds_label=label,
                    preds_prob=prob
                )
            else:
                flash(u'Image does not exists')
                return redirect(request.url)
    else:
        return render_template("predict.html")        


if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port='8080')