from flask import render_template
from src.app import app


@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html')


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html')
