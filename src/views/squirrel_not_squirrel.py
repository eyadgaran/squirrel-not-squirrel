from flask import render_template
from src.app import app


@app.route('/')
@app.route('/index')
@app.route('/squirrel_not_squirrel')
def squirrel_not_squirrel():
    return render_template('pages/squirrel_not_squirrel.html')


@app.route('/about')
def about():
    return render_template('pages/about.html')
