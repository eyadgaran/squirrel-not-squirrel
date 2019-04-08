'''
Module for "which squirrel" endpoint
'''

from quart import render_template


async def which_squirrel():
    return await render_template('pages/which_squirrel.html')
