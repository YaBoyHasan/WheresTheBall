from flask import Flask
from config import Config
from .core.routes import core_blueprint
from .auth.routes import auth_blueprint

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.register_blueprint(core_blueprint)
    app.register_blueprint(auth_blueprint)
    return app