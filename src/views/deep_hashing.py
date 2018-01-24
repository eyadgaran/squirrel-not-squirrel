from flask import request, render_template, flash, redirect, url_for
from src.app import app


@app.route('/which_squirrel')
def which_squirrel():
    return render_template('pages/which_squirrel.html')



