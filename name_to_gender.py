'''
A class-based gender-guesser based on the messy older gender_inference.py
'''

from pathlib import Path
import json
from nameparser import HumanName

from dataset_dissertation import DissertationDataset

import gender_guesser.detector

class GenderGuesser:

    def __init__(self):

        self._stanford_name_to_gender_dict = self.get_stanford_name_to_gender_dict()
        self._gender_guesser_instance = gender_guesser.detector.Detector()
        self._gender_of_name_assigned_by_hand_dict = self.get_hand_assigned_names()

    def guess_gender_of_first_name(self, name: str):
        """
        Guesses gender of a name


        >>> gg = GenderGuesser()
        >>> gg.guess_gender_of_first_name('Mary')
        'female'
        >>> gg.guess_gender_of_first_name('Shirley')
        'female'
        >>> gg.guess_gender_of_first_name('L.')
        'unknown'

        :return:
        """

        if name in self._gender_of_name_assigned_by_hand_dict:
            return self._gender_of_name_assigned_by_hand_dict[name]
        else:
            gen_usa = self._gender_guesser_instance.get_gender(name, 'usa')
            gen_all = self._gender_guesser_instance.get_gender(name)

            if gen_usa in {'female', 'male'}:
                gender = gen_usa

            # applies to e.g. George, Jerry, Christian, Max, Gabriel
            elif gen_usa == 'mostly_male' and gen_all == 'male':
                gender = 'male'

            # applies mostly to international names
            # Jorge, Jaime*, Eduardo, Pablo, Mohamed, Fernando,
            elif gen_usa == 'andy' and gen_all == 'male':
                gender = 'male'

            # applies to Sharon, Dana, Shirley, Carmen, Clare, Sandie
            elif gen_usa == 'mostly_female' and gen_all == 'female':
                gender = 'female'

            # applies mostly to international names
            # Ana, Karin, Elisabeth, Marta, Elena, Marguerite, Olga
            elif gen_usa == 'andy' and gen_all == 'female':
                gender = 'female'

            else:
                gender = gen_usa

            return gender

    def guess_gender_including_international_names(self, human_name: HumanName):
        """
        Guesses the gender of a complete HumanName object
        Uses first and middle name and both u.s. and international dictionaries.

        >>> gg = GenderGuesser()
        >>> h = HumanName('Perez, Haven Abraham')
        >>> gg.guess_gender_including_international_names(h)
        'male'

        # this function uses the middle name to get additional information compared to just using
        # the first name.
        >>> h = HumanName('Perez, Haven')
        >>> gg.guess_gender_of_first_name(h.first)
        'andy'

        :param human_name: HumanName
        :return: str
        """

        # Turn "Marten L." into "Marten
        if human_name.middle:
            middle = human_name.middle.split()[0]
        else:
            middle = None

        gender_first = self.guess_gender_of_first_name(human_name.first)

        if gender_first in {'male', 'female'}:
            return gender_first

        # if middle name exists and is not initial ("A."), examine it
        elif middle and (len(middle) > 2 or middle[-1] != '.'):

            gender_middle = self.guess_gender_of_first_name(middle)

            # if middle name doesn't give us any new info -> return first name info
            if gender_middle in ['unknown', 'andy']:
                return gender_first

            elif gender_middle == 'male' and gender_first in ['mostly_male', 'andy', 'unknown']:
                return 'male'
            elif gender_middle == 'female' and gender_first in ['mostly_female', 'andy', 'unknown']:
                return 'female'

            else:
                if gender_middle in ['male', 'female']:
                    print(f'{human_name.first} {human_name.middle} {human_name.last}', gender_first,
                          gender_middle)
                return gender_first

        else:
            return gender_first

    def guess_gender_stanford(self, human_name: HumanName):
        """
        Returns the gender guessed by the model trained by Dan's group at Stanford
        (uses the dissertation dataset for inference)

        >>> h = HumanName('Perez, Ali Abraham')
        >>> gg = GenderGuesser()
        >>> gg.guess_gender_stanford(h)
        'female'

        :param human_name: HumanName
        :return: str
        """

        if len(human_name.first) == 1:
            name_to_guess = human_name.middle
        elif len(human_name.first) == 2 and human_name.first[1] == '.':
            name_to_guess = human_name.middle
        else:
            name_to_guess = human_name.first

        if human_name.first in self._stanford_name_to_gender_dict:
            return self._stanford_name_to_gender_dict[name_to_guess]
        else:
            return 'unknown'

    def guess_gender_census(self, human_name: HumanName, return_type='gender'):
        """
        Use census data to infer gender.

        >>> gg = GenderGuesser()
        >>> h = HumanName('Perez, Ali Abraham')
        >>> gg.guess_gender_census(h, return_type='gender')
        'andy'

        # the function can also be used to get the probabily that a name is male in the census
        >>> gg.guess_gender_census(h, return_type='probability_male')
        '72% male'

        >>> h = HumanName('Perez, A. Abraham')
        >>> gg.guess_gender_census(h)
        'male'

        >>> h = HumanName('Perez, A Abraham')
        >>> gg.guess_gender_census(h)
        'male'

        :param human_name:
        :param return_type:
        :return:
        """

        if len(human_name.first) == 1:
            name_to_guess = human_name.middle.lower()
        elif len(human_name.first) == 2 and human_name.first[1] == '.':
            name_to_guess = human_name.middle.lower()
        else:
            name_to_guess = human_name.first.lower()

        if name_to_guess in CENSUS_NAME_TO_MALE_PROBABILITY:
            male_prob = CENSUS_NAME_TO_MALE_PROBABILITY[name_to_guess]
            if male_prob < 5:
                inferred_gender = 'female'
            elif male_prob < 15:
                inferred_gender = 'mostly_female'
            elif male_prob < 85:
                inferred_gender = 'andy'
            elif male_prob < 95:
                inferred_gender = 'mostly_male'
            else:
                inferred_gender = 'male'

            if return_type == 'gender':
                return inferred_gender
            elif return_type == 'probability_male':
                return '{:2.0f}% male'.format(male_prob)

        else:
            return 'unknown'


    @staticmethod
    def get_hand_assigned_names():
        """
        Adds some hand-coded names to the gender-guesser library

        :return: dict
        """
        return {
            # mostly female to female
            'Mary': 'female',
            'Carol': 'female',
            'Sharon': 'female',
            'Kimberly': 'female',
            'Erin': 'female',
            'Lauren': 'female',
            'Shirley': 'female',
            'Meredith': 'female',
            'Carmen': 'female',
            'Courtney': 'female',

            # male to andy
            'Alex': 'andy',
            'Ali': 'andy',
            'Micah': 'andy',
            'Jean': 'andy',

            # male USA, female international (or vice versa) -> andy
            'Marian': 'andy',
            'Patrice': 'andy',
            'Toni': 'andy',
            'Laurence': 'andy',
            'Eli': 'andy',
            'Sasha': 'andy',
            'Lon': 'andy',

            # male to andy
            'Jaime': 'andy',
            'Glen': 'andy',
            'Rory': 'andy',
            'Jess': 'andy',
            'Ricki': 'andy',

            # mostly male to andy
            'Whitney': 'andy',
            'Leslie': 'andy',

            # female to andy
            'Haven': 'andy',
            'Shannan': 'andy',

            # mostly female to andy
            'Robin': 'andy',

            # mostly male to male
            'George': 'male',
            'Ryan': 'male',
            'Jerry': 'male',
            'Kyle': 'male',
            'Christian': 'male',
        }




    @staticmethod
    def get_stanford_name_to_gender_dict():

        try:
            with open(Path('data', 'stanford_name_to_gender.json'), 'r') as infile:
                 return json.load(infile)
        except FileNotFoundError:
            d = DissertationDataset()
            stanford_name_to_gender_dict = {}
            for _, row in d.df.iterrows():
                name = row.AdviseeID.split(':')[0].replace('_', ' ').replace('-', ' ')
                name = " ".join([x.capitalize() for x in name.split()])
                human_name = HumanName(name)
                guess_proquest = row['AdviseeGender.1']
                stanford_name_to_gender_dict[human_name.first] = guess_proquest
            # store dict of stanford names to gender
            with open(Path('data', 'gender_inference', 'stanford_name_to_gender.json'), 'w') as out:
                json.dump(stanford_name_to_gender_dict, out, sort_keys=True, indent=4)
            return stanford_name_to_gender_dict
