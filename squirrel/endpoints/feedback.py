'''
Module for "feedback" endpoints
'''

__author__ = 'Elisha Yadgaran'


from quart import render_template, redirect, url_for, flash, request
from squirrel.database.models import Feedback


async def feedback():
    return await render_template('pages/feedback.html')


async def record_feedback():
    form = await request.form
    user_feedback = form['feedback']
    Feedback.create(feedback=user_feedback)
    await flash("Your feedback was recorded. Thank You!")
    return redirect(url_for('squirrel_not_squirrel'))
