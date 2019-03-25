'''
Module for "which squirrel" endpoint
'''

from flask import render_template


def which_squirrel():
    return render_template('pages/which_squirrel.html')
