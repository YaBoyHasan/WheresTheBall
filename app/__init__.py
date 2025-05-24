from flask import Flask
from .core.routes import core_blueprint

def create_app():
    app = Flask(__name__)
    app.secret_key = "dev-key"
    app.register_blueprint(core_blueprint)
    return app