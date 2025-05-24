from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from .forms import LoginForm, RegisterForm
from app.models.user import User, db
from werkzeug.security import generate_password_hash

auth_blueprint = Blueprint("auth", __name__, url_prefix="/auth")

@auth_blueprint.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            session["user_id"] = user.id
            flash("Logged in successfully", "success")
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
    form = RegisterForm()
    if form.validate_on_submit():
        # Invite code check
        if form.invite_code.data != "SECRET123":
            flash("Invalid invite code", "danger")
            return redirect(url_for("auth.register"))

        # Check if user already exists
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash("Email already registered", "warning")
            return redirect(url_for("auth.login"))

        # Create new user
        new_user = User(email=form.email.data)
        new_user.set_password(form.password.data)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html", form=form)