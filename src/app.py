from flask import Flask
from src.database.initialization import SimpleMLDatabase, AppDatabase


class FlaskConfig(object):
    # Secret key for session management. You can generate random strings here:
    # http://clsc.net/tools-old/random-string-generator.php
    SECRET_KEY = 'squirrel-nado'
    # Flask-WTF flag for CSRF
    CSRF_ENABLED = True


# template_dir = os.path.abspath('../templates')
app = Flask(__name__)
app.config.from_object(FlaskConfig)


if __name__ == '__main__':
    SimpleMLDatabase().initialize()
    AppDatabase().initialize()

    from views.modeling import *
    from views.errors import *
    from views.squirrel_not_squirrel import *
    from views.deep_hashing import *
    from views.feedback import *

    print app.url_map
    app.run(host='0.0.0.0', port=6800, debug=False)
