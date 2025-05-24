from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from app.forms.auth import LoginForm, RegisterForm
from app.models.user import User, db
from werkzeug.security import generate_password_hash
from datetime import datetime

auth_blueprint = Blueprint("auth", __name__, url_prefix="/auth")

@auth_blueprint.route("/login", methods=["GET", "POST"])
def login():
    if 'user_id' in session:
        return redirect(url_for('core.home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            session['user_id'] = user.id
            user.last_login = datetime.utcnow()
            user.last_ip = request.remote_addr
            db.session.commit()
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for("core.home"))
        else:
            flash("Invalid email or password", "danger")
    return render_template("auth/login.html", form=form)

@auth_blueprint.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out", "info")
    return redirect(url_for("core.home"))

@auth_blueprint.route("/register", methods=["GET", "POST"])
def register():
    if 'user_id' in session:
        return redirect(url_for('core.home'))
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter(
            (User.email == form.email.data) | (User.username == form.username.data)
        ).first()
        if existing_user:
            flash("Email or Username already taken", "warning")
            return redirect(url_for("auth.register"))

        # Inside your registration route
        user = User(
            username=form.username.data,
            email=form.email.data,
            registered_at=datetime.utcnow(),
            last_ip=request.remote_addr
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html", form=form)
