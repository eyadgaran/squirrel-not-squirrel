'''
Module for "Home" endpoint
'''

__author__ = 'Elisha Yadgaran'


from quart import render_template


async def squirrel_not_squirrel():
    return await render_template('pages/squirrel_not_squirrel.html')
