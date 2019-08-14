
from pathlib import Path
import pandas as pd

def load_topic_data():
    topics = {}
    df = pd.read_csv(Path('data','topic_terms.csv'))
    for _, row in df.iterrows():
        topic_id = int(row['Topic'])
        terms_prob = row['prob'].split(", ")
        terms_frex = row['frex'].split(", ")
        topics[topic_id] = {
            'name': row['Topic Labels'],
            'terms_prob': terms_prob,
            'terms_frex': terms_frex,
            'terms_both': terms_prob + terms_frex
        }
    return topics

TOPICS = load_topic_data()

# TOPIC_IDS_TO_NAME = {
#     1: 'Journalism',
#     2: 'US Oil  / Regional (Oklahoma)',
#     3: 'Criminal Justice System',
#     4: 'Labor',
#     5: 'Eastern Europe',
#     6: 'US Political History; US Presidents',
#     7: 'Panama Canal',
#     8: 'Rural',
#     9: 'Latin America',
#     10: 'Civil rights',
#     11: 'Medieval',
#     12: 'Ottoman empire',
#     13: 'US Progressive Era ',
#     14: 'Hispanic ',
#     15: 'Art/ African American Art/Dance',
#     16: 'Islamic ',
#     17: 'Quantitative Methods',
#     18: 'Native american',
#     19: 'Literary',
#     20: 'Violence',
#     21: 'Roman empire',
#     22: 'Cultural - gender/class',
#     23: 'Economic/Business',
#     24: 'Science',
#     25: 'Political Parties',
#     26: 'Labor',
#     27: 'Industry',
#     28: 'Gender',
#     29: 'Urban',
#     30: 'World War II: Pacific Theater',
#     31: 'French revolution',
#     32: 'Colonialism',
#     33: 'Education',
#     34: 'Church',
#     35: 'Family/Household',
#     36: 'Medical',
#     37: 'Jewish',
#     38: 'Slavery',
#     39: 'History of the West',
#     40: 'Civil War',
#     41: 'East Asia',
#     42: 'Intellectual',
#     43: 'Development',
#     44: 'British History',
#     45: 'Music',
#     46: 'History of the West',
#     47: 'Classics',
#     48: 'Sexuality',
#     49: 'Labor',
#     50: 'Agrarian/Rural',
#     51: 'legal',
#     52: 'Central and Eastern Europe/ 20th Century',
#     53: 'GENERIC',
#     54: 'Sports',
#     55: 'Military',
#     56: 'Civil Rights',
#     57: 'New England',
#     58: 'African American',
#     59: 'Film/Art',
#     60: 'Italian Renaissance',
#     61: 'Cultural',
#     62: 'GENERIC',
#     63: 'Cultural - Identity construction',
#     64: 'Christian',
#     65: 'Biography ; US Presidents',
#     66: 'Political ',
#     67: 'International Relations/Latin America',
#     68: 'Political; Government',
#     69: 'Noise; instructions for filing diss',
#     70: 'Noise'
# }
#
