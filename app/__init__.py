from flask import Flask, g, session
from config import Config
from app.routes.core import core_blueprint
from app.routes.auth import auth_blueprint
from app.models.user import db, User
from apscheduler.schedulers.background import BackgroundScheduler
from app.utils.scraper import fetch_and_store_comp_data  # adjust path if needed
import os
import logging

logging.getLogger('apscheduler').setLevel(logging.DEBUG)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    @app.cli.command("train-vanillacnn")
    def train_vanillacnn():
        from app.predictors.VanillaCNN import train
        train.run_training()

    @app.cli.command("train-mobilenetdense")
    def train_vanillacnn():
        from app.predictors.MobileNetDense import train
        train.run_training()

    @app.cli.command("train-efficientnet")
    def train_vanillacnn():
        from app.predictors.EfficientNet import train
        train.run_training()

    @app.cli.command("train-channelattentionnet")
    def train_vanillacnn():
        from app.predictors.ChannelAttentionNet import train
        train.run_training()

    @app.cli.command("train-unetheatmap")
    def train_vanillacnn():
        from app.predictors.UNetHeatmap import train
        train.run_training()

    @app.before_request
    def load_logged_in_user():
        user_id = session.get("user_id")
        if user_id:
            g.user = User.query.get(user_id)
        else:
            g.user = None

    app.register_blueprint(core_blueprint)
    app.register_blueprint(auth_blueprint)

   # APScheduler setup
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        scheduler = BackgroundScheduler()

        def scheduled_job():
            with app.app_context():
                fetch_and_store_comp_data()

        scheduler.add_job(func=scheduled_job, trigger="interval", hours=1)
        scheduler.start()

    return app
