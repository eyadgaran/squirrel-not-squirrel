import os
import zipfile
from src.config import parse_cnf
import psycopg2


def extract_data(zip_file, target_idr):
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    if not os.path.exists(target_idr):
        os.mkdir(target_idr)
    zip_ref.extractall(target_idr)
    zip_ref.close()


def import_squirrels(description_file):
    params = parse_cnf('local-squirrel')
    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    with open(description_file) as f:
        for line in f:
            filename, description = line.strip().split('\t')
            cur.execute("INSERT INTO squirrel_descriptions (filename, description) VALUES (%s, %s)", (filename, description))
    conn.commit()
    cur.close()
    conn.close()
    print "done inserting descriptions!"


if __name__ == "__main__":
    extract_data('../../squirrels.zip', '../static/')
    import_squirrels('../static/squirrels/descriptions.txt')
