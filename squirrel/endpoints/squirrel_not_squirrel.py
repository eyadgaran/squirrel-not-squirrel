'''
Module for "Home" endpoint
'''

__author__ = 'Elisha Yadgaran'


from flask import render_template


def squirrel_not_squirrel():
    return render_template('pages/squirrel_not_squirrel.html')