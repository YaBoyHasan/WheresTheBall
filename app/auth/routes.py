from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from .forms import LoginForm, RegisterForm

auth_blueprint = Blueprint("auth", __name__, url_prefix="/auth")

@auth_blueprint.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Dummy login logic — replace with real DB check later
        if form.email.data == "admin@example.com" and form.password.data == "password":
            session["user_id"] = 1
            session["email"] = form.email.data
            flash("Logged in successfully", "success")
            return redirect(url_for("core.home"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("auth/login.html", form=form)

@auth_blueprint.route("/logout")
def logout():
    session.clear()
    flash("You’ve been logged out.", "info")
    return redirect(url_for("core.home"))

@auth_blueprint.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        # ⚠ Replace with actual DB insert logic
        # Fake registration: save email/pass in session (temp)
        session["user_id"] = email  # Replace with user ID from DB in real app
        flash("Registration successful!", "success")
        return redirect(url_for("core.home"))

       # if form.invite_code.data != "SECRET123":
       # flash("Invalid invite code", "danger")
       # return redirect(url_for("auth.register"))

    return render_template("auth/register.html", form=form)