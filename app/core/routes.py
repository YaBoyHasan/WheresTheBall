from flask import Blueprint, render_template

core_blueprint = Blueprint("core", __name__)

latest_prediction = {
    "comp_name": "BOTB Week 42",
    "x": 58,
    "y": 72,
    "image_url": "images/sample.jpeg"  # Make sure this exists in static/images/
}

@core_blueprint.route("/")
def home():
    return render_template("home.html")

@core_blueprint.route("/about")
def about():
    return render_template("about.html")

@core_blueprint.route("/past")
def past_predictions():
    return render_template("past.html")

@core_blueprint.route("/latest")
def latest_prediction():
    return render_template("latest.html", latest=latest_prediction)