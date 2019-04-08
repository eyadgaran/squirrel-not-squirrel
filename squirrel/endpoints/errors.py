'''
Module for "error" endpoints
'''

__author__ = 'Elisha Yadgaran'


from quart import render_template


async def internal_error(error):
    #db_session.rollback()
    return await render_template('errors/500.html')


async def not_found_error(error):
    return await render_template('errors/404.html')
