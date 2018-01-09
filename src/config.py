import ConfigParser
from os.path import expanduser


def parse_cnf(cnf_section):
    '''
    Assumes there is a cnf file located at ~/.my.cnf with sections and parameters
    :param cnf_section: cnf section title for param group
    :return: dictionary of parameters (None defaults to empty string)
    '''
    config = ConfigParser.SafeConfigParser(allow_no_value=True)
    config.read(expanduser("~/.my.cnf"))
    parameter_dict = dict(config.items(cnf_section))

    return parameter_dict


def create_connection(cnf_section):
    '''
    Assumes there is a cnf file located at ~/.my.cnf with sections and parameters
    Required section parameters: user, password, host, port, database
    :param cnf_section: cnf section title for param group
    :return: sqlalchemy connection engine
    '''
    parameter_dict = parse_cnf(cnf_section)

    url = 'postgresql://{user}:{password}@{host}:{port}/{database}'.format(
        **parameter_dict)

    return url


class FlaskConfig(object):
    # Secret key for session management. You can generate random strings here:
    # http://clsc.net/tools-old/random-string-generator.php
    SECRET_KEY = 'squirrel-nado'
    SQLALCHEMY_DATABASE_URI = create_connection('local-squirrel')
    UPLOADED_PHOTOS_DEST = 'static/uploads'
    # Flask-WTF flag for CSRF
    CSRF_ENABLED = True
