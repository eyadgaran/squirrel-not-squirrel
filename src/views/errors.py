from flask import render_template


def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html')


def not_found_error(error):
    return render_template('errors/404.html')
