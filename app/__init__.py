from flask import Flask, g, session
from config import Config
from app.routes.core import core_blueprint
from app.routes.auth import auth_blueprint
from app.models.user import db, User

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    @app.before_request
    def load_logged_in_user():
        user_id = session.get("user_id")
        if user_id:
            g.user = User.query.get(user_id)
        else:
            g.user = None

    app.register_blueprint(core_blueprint)
    app.register_blueprint(auth_blueprint)

    return app
