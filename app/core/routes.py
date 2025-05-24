from flask import Blueprint, render_template

latest_prediction = {
    "comp_name": "BOTB Week 42",
    "x": 58,
    "y": 72,
    "image_url": "images/sample.jpeg"  # Get CompName, Predicted(X,Y), Image dynamically later
}

# core area
core_blueprint = Blueprint("core", __name__)

@core_blueprint.route("/")
def home():
    return render_template("core/home.html")

@core_blueprint.route("/about")
def about():
    return render_template("core/about.html")

@core_blueprint.route("/past")
def past_predictions():
    return render_template("core/past.html")

@core_blueprint.route("/latest")
def latest_prediction():
    return render_template("core/latest.html", latest=latest_prediction)

# auth area
#   auth_blueprint = Blueprint("auth", __name__)