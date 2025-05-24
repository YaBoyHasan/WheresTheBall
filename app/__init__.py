from flask import Flask
from config import Config
from .core.routes import core_blueprint

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.register_blueprint(core_blueprint)
    return app