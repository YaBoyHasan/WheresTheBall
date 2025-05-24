from flask import Flask
from config import Config
from app.core.routes import core_blueprint
from app.auth.routes import auth_blueprint
from app.models.user import db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    db.init_app(app)

    with app.app_context():
        db.create_all()

    app.register_blueprint(core_blueprint)
    app.register_blueprint(auth_blueprint)

    return app
