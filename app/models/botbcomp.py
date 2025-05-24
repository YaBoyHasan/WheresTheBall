from app.models.user import db  # reuse your existing db object
from datetime import datetime

class BotbComp(db.Model):
    __tablename__ = 'BotbComps'

    id = db.Column(db.Integer, primary_key=True)
    CompUrl = db.Column(db.String(255), unique=True, nullable=False)
    JudgesX = db.Column(db.Integer, nullable=True)
    JudgesY = db.Column(db.Integer, nullable=True)
    ImageFileName = db.Column(db.String(255), nullable=True)