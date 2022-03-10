import tensorflow as tf
import os
import pandas as pd
from config import app
from flask import request, render_template, flash, redirect, session
from predict import is_image_exist_google, predict_image, get_image_google, get_image_osc, load_model, allowed_file, save_csv

model = load_model(app.config["MODEL_PATH"])

@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        longitude = str(request.form['Longitude'])
        latitude = str(request.form['Latitude'])
        api_options = str(request.form['api_options'])
        save_dir = app.config['IMG_FOLDER']

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
            image_path = get_image_osc(query_params, save_dir)
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


@app.route('/predict_csv', methods=["GET", "POST"])
def predict_csv():
    if request.method == "POST":
        if 'csv_file' not in request.files:
            flash(u'No file uploaded')
            return redirect(request.url)

        file = request.files['csv_file']
        api_options = str(request.form['api_options'])
        
        if file and allowed_file(file.filename):
            csv_path = save_csv(file)

        building_df = pd.read_csv(csv_path, delimiter=',')
        building_result = []
        for index, row in building_df.iterrows():
            latitude = row['lat']
            longitude = row['lon']


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
                image_np = get_image_google(query_params, csv=True)
                _ , label, prob = predict_image(model, img_np=image_np)
                
            elif api_options == "osc":
                query_params = {
                    "lng": longitude,
                    "lat": latitude,
                    "projection": "PLANE",
                    "projectionYaw": "0"
                }
                image_np = get_image_osc(query_params, csv=True)
                if image_np is not None:
                    _ , label, prob = predict_image(model, img_np=image_np)

            building_row = [latitude, longitude, label, prob]
            building_result.append(building_row)

        return render_template("predict_csv.html", building_result=building_result)
    else:
        return render_template("predict_csv.html")           


if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port='8080')