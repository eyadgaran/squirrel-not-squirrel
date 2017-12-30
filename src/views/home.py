from flask import render_template
from src.app import app


@app.route('/')
@app.route('/index')
def home():
    return render_template('pages/home.html')


@app.route('/about')
def about():
    return render_template('pages/about.html')
