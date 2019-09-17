import sqlite3
from pathlib import Path
from IPython import embed
import pandas

from collections import defaultdict

def generate_journal_csv():
    """
    Uses the JSTOR_full_cleaned database to generate a csv that lists the number of research articles
    by journal and decade from 1940 to the present, english only.

    :return:
    """

    journal_data = defaultdict(defaultdict)

    db = sqlite3.connect(Path('data', 'JSTOR_full_cleaned.db'))
    cur = db.cursor()
    cur.execute('''select journal, year from article_pub_info 
                      where article_type="research article" and year > 1940 and language="eng";''')
    while True:
        row = cur.fetchone()
        if not row:
            break

        journal = row[0]
        # 1988 -> 1980
        decade = int(row[1]) // 10 * 10

        try:
            journal_data[journal][decade] += 1
        except KeyError:
            journal_data[journal][decade] = 1

    df = pandas.DataFrame(journal_data).fillna(0)
    df = df.sort_index().transpose()
    df['total'] = df[1940] + df[1950] + df[1960] + df[1970] + df[1980] + df[1990] + df[2000] + df[2010]
    df = df.sort_values(by='total', ascending=False)

    df.to_csv(Path('data', 'journals_by_decade.csv'))

    embed()


if __name__ == '__main__':
    generate_journal_csv()