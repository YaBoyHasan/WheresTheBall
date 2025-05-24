from flask import Blueprint, render_template, url_for

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
    latest = {
        "comp_name": "BOTB Week 42",
        "x": 1258,
        "y": 3272,
        "image_url": url_for('static', filename='images/prediction.jpeg')  # Get CompName, Predicted(X,Y), Image dynamically later
    }
    return render_template("core/latest.html", latest=latest)