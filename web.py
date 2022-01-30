import tensorflow as tf
from config import app
from flask import request, render_template, flash, redirect
from predict import is_metadata_exist, predict_image, get_image, load_model

model = load_model(app.config["MODEL_PATH"])

@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        longitude = str(request.form['longitude'])
        latitude = str(request.form['latitude'])
        API_KEY = app.config["API_KEY"]
        query_params = {
            "location": f"{latitude},{longitude}",
            "size": "224x224",
            "key": API_KEY,
            "return_error_code": "true",
            "source": "outdoor",
            "radius": 10
        }

        if is_metadata_exist(query_params) == True:
            save_dir = app.config['IMG_FOLDER']
            image_path = get_image(query_params, save_dir)
            class_id, label, prob = predict_image(model, image_path)
            return render_template(
                "predict.html",
                img_path=image_path,
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