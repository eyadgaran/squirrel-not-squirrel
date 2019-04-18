from quart import Quart
from squirrel.database.initialization import SimpleMLDatabase, AppDatabase
from squirrel import TEMPLATE_PATH, STATIC_PATH


class FlaskConfig(object):
    # Secret key for session management. You can generate random strings here:
    # http://clsc.net/tools-old/random-string-generator.php
    SECRET_KEY = ''
    # Flask-WTF flag for CSRF
    CSRF_ENABLED = True


app = Quart(__name__, template_folder=TEMPLATE_PATH, static_folder=STATIC_PATH)
app.config.from_object(FlaskConfig)
SimpleMLDatabase().initialize()
AppDatabase().initialize()

from squirrel.endpoints.deep_hashing import which_squirrel
from squirrel.endpoints.errors import internal_error, not_found_error
from squirrel.endpoints.feedback import feedback, record_feedback
from squirrel.endpoints.modeling import upload, model_feedback
from squirrel.endpoints.squirrel_not_squirrel import squirrel_not_squirrel

app.register_error_handler(500, internal_error)
app.register_error_handler(404, not_found_error)

app.add_url_rule('/', '', squirrel_not_squirrel)
app.add_url_rule('/index', 'index', squirrel_not_squirrel)
app.add_url_rule('/squirrel_not_squirrel', 'squirrel_not_squirrel', squirrel_not_squirrel)
app.add_url_rule('/which_squirrel', 'which_squirrel', which_squirrel)
app.add_url_rule('/feedback', 'feedback', feedback, methods=['GET'])
app.add_url_rule('/record_feedback', 'record_feedback', record_feedback, methods=['POST'])
app.add_url_rule('/upload/<feature>', 'upload', upload, methods=['GET', 'POST'])
app.add_url_rule('/record_model_feedback', 'model_feedback', model_feedback, methods=['POST'])

if __name__ == '__main__':
    # print(app.url_map.endpoints)
    app.run(host='127.0.0.1', port=8081, debug=False)
