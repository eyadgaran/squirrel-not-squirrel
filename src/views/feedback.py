from flask import render_template, redirect, url_for, flash, request
from src.database.models import Feedback


def feedback():
    return render_template('pages/feedback.html')


def record_feedback():
    user_feedback = request.form['feedback']
    Feedback.create(feedback=user_feedback)
    flash("Your feedback was recorded. Thank You!")
    return redirect(url_for('squirrel_not_squirrel'))
