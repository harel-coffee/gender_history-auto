# This file contains functions to parse the metadata from the xml files into a sqlite database


import json
import os
import sqlite3
from pathlib import Path
from time import strptime

import gender_guesser.detector
from IPython import embed
from bs4 import BeautifulSoup

from database import initialize_metadata_db

GENDER_GUESSER = gender_guesser.detector.Detector()


def parse_xml_metadata_of_all_articles():
    """
    Parses all xml files in the data/metadata folder and stores the metadata in the
    metadata.db sqlite db

    :return:
    """


    metadata_path = Path('data', 'metadata')
    initialize_metadata_db()
    db = sqlite3.connect(str(Path('data', 'metadata.db')))
    cur = db.cursor()

    count = 0
    articles = []
    for root, _, files in os.walk(metadata_path):
        for file in files:
            if file.endswith('.xml'):
                print(count)
                count += 1
                article = parse_article_metadata_from_xml(Path(root, file))
                if article:
                    articles.append(article)

    # insert articles into db sorted by date asc
    for article_id, article in enumerate(sorted(articles, key = lambda x:x['date_str'])):
        article['article_id'] = article_id
        cur.execute('''REPLACE INTO metadata (article_id, uri, doi, jstor_id, filename, title,
                                             authors_list, authors_str, authors_gender, authors_number,
                                             year, date_str, journal_title, volume, issue, pages)
                                     VALUES (:article_id, :uri, :doi, :jstor_id, :filename, :title,
                                             :authors_list, :authors_str, :authors_gender, :authors_number,
                                             :year, :date_str, :journal_title, :volume, :issue, :pages);
                                             ''', article)
    db.commit()


def parse_article_metadata_from_xml(file_path):
    """
    Parses the metadata from an xml an inserts it into the metadata database
    Note: for the time being, it skips all articles before 1970 because they tend to be harder
    to parse, e.g. authors are just initials or absent.


    :param file_path
    :return:
    """
    print('\n', file_path)

    with open(file_path) as f:
        soup = BeautifulSoup(f, "lxml-xml")

    article = {
        'filename': file_path.name[:-4]
    }
    article_meta = soup.front.select('article-meta')[0]

    # Identifiers
    article['uri'] = soup.find_all('self-uri')[0]['xlink:href']

    # jstor id is sometimes directly available. But sometimes needs to be extracted from URI
    if article_meta.find_all('article-id', {'pub-id-type' : 'jstor'}):
        article['jstor_id'] = article_meta.find_all('article-id', {'pub-id-type' : 'jstor'})[0].text
    elif article_meta.find_all('article-id', {'pub-id-type' : 'jstor-stable'}):
        article['jstor_id'] = article_meta.find_all('article-id', {'pub-id-type' : 'jstor-stable'})[0].text
    else:
        article['jstor_id'] = 'not available'

    try:
        article['doi'] = article_meta.find('article-id', {'pub-id-type': 'doi'}).text
    except AttributeError:
        article['doi'] = 'not available'

    # Title
    article['title'] = article_meta.find_all('article-title')[0].text

    # Date
    article['year'] = int(article_meta.select('pub-date')[0].year.text)
    # TODO: Parse articles before 1970
    if article['year'] < 1970:
        print("skipping, year: ", article['year'])
        return

    # day of pub is sometimes available, otherwise set to 1.
    try:
        day = int(article_meta.select('pub-date')[0].day.text)
    except AttributeError:
        day = 1

    # some months are numbers, others are strings ("February")
    month = article_meta.select('pub-date')[0].month.text
    try:
        month = int(month)
    except ValueError:
        month = int(strptime(month, '%B').tm_mon)

    article['date_str'] = '{:04d}-{:02d}-{:02d}'.format(article['year'], month, day)

    # Journal Info
    title_group = soup.front.select('journal-meta')[0].select('journal-title-group')[0]
    article['journal_title'] = title_group.select('journal-title')[0].text
    article['volume'] = int(article_meta.volume.text)
    article['issue'] = int(article_meta.issue.text)
    article['pages'] = f"{article_meta.select('fpage')[0].text}-{article_meta.select('lpage')[0].text}"

    # Authors
    authors_gender, authors_list, authors_str, authors_number = parse_contributors(article_meta)
    article['authors_list'] = authors_list
    article['authors_str'] = authors_str
    article['authors_gender'] = authors_gender
    article['authors_number'] = authors_number

    return article



def parse_contributors(article_meta):

    contribs = article_meta.select('contrib-group')

    # some authors, e.g. minutes, have no authors listed.
    if not contribs:
        authors_gender = 'undetermined'
        authors_list = json.dumps([])
        authors_str = ''
        authors_number = 0
        return authors_gender, authors_list, authors_str, authors_number

    else:
        contribs = contribs[0]
        authors_list = []

        for author in contribs.find_all('contrib'):
            last_name, given_name, gender = parse_contributor(author)

            authors_list.append({
                'last': last_name,
                'given': given_name,
                'gender': gender
            })

        # Determine overall gender of author or authors
        genders = set([x['gender'] for x in authors_list])
        # one or multiple authors, all male
        if genders == {'male'}:
            authors_gender = 'male'
        # ... or all female
        elif genders == {'female'}:
            authors_gender = 'female'
        # ... or at least one male and one female
        elif genders == {'female', 'male'}:
            authors_gender = 'mixed'
        # ... or everything else, including androgynous or unknown names
        else:
            authors_gender = 'undetermined'
            print(f"author genders undetermined for {authors_list}.")


        authors_str = " ".join([f"{x['last']}, {x['given']}" for x in authors_list])
        authors_number = len(authors_list)
        authors_list = json.dumps(authors_list)

        return authors_gender, authors_list, authors_str, authors_number

def parse_contributor(contrib_xml):
    """
    Parses last name, first name, and gender of one contributor

    :param contrib_xml:
    :return:
    """

    author = contrib_xml.select('string-name')[0]

    # sometimes first and last names are under separate sub-tags. Sometimes they aren't because
    # logic
    if len(author.select('given-names')) > 0:
        given_name = author.select('given-names')[0].text
        last_name = author.select('surname')[0].text.capitalize()

    # e.g. "Arthur C. Cole"
    else:
        given_name = " ".join(author.text.split()[:-1])
        last_name = author.text.split()[-1]

    # names can come in ALL CAPS -> capitalize
    given_name = " ".join([x.capitalize() for x in given_name.split()])
    last_name = last_name.capitalize()

    # run gender guesser on the "main name", usually the first name but not always
    # e.g. for "H. Nelson" we want to run it on "Nelson" not "H."
    main_name = given_name.split()[0]
    if main_name[1] == '.' and len(given_name.split()) > 1:
        main_name = given_name.split()[1]
    gender = GENDER_GUESSER.get_gender(main_name, 'usa')
    print(given_name, main_name, gender)

    return last_name, given_name, gender



if __name__ == '__main__':
    parse_xml_metadata_of_all_articles()