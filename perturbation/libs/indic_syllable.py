from dataclasses import dataclass
from collections import Counter, defaultdict
import youseedee as ysd
import re

INDIC_LANGS = [
    'Ahom', 'Balinese', 'Batak', 'Bengali', 'Bhaiksuki', 'Brahmi', 'Buginese', 'Buhid',
    'Chakma', 'Cham', 'Devanagari', 'Dives Akuru', 'Dogra', 'Grantha', 'Gujarati',
    'Gunjala Gondi', 'Gurmukhi', 'Hanunoo', 'Javanese', 'Kaithi', 'Kannada',
    'Kayah Li', 'Kharoshthi', 'Khmer', 'Khojki', 'Khudawadi', 'Lao', 'Lepcha', 'Limbu',
    'Mahajani', 'Makasar', 'Malayalam', 'Marchen', 'Masaram Gondi', 'Meetei Mayek',
    'Modi', 'Multani', 'Myanmar', 'Nandinagari', 'Newa', 'New Tai Lue', 'Oriya',
    'Phags-pa', 'Rejang', 'Saurashtra', 'Sharada', 'Siddham', 'Sinhala', 'Soyombo',
    'Sundanese', 'Syloti Nagri', 'Tagalog', 'Tagbanwa', 'Tai Le', 'Tai Tham',
    'Tai Viet', 'Takri', 'Tamil', 'Telugu', 'Thai', 'Tibetan', 'Tirhuta',
    'Zanabazar Square'
]

INVISIBLE_CHARS = [
    u'\u00a0', u'\u034f', u'\u061c', u'\u115f', u'\u1160', u'\u17b4', u'\u17b5', 
    u'\u180e', u'\u2000', u'\u2001', u'\u2002', u'\u2003', u'\u2004', u'\u2005', u'\u2006',
    u'\u2007', u'\u2008', u'\u2009', u'\u200a', u'\u200b', u'\u200c', u'\u200d', u'\u200e', 
    u'\u200f', u'\u202f', u'\u205f', u'\u2060', u'\u2061', u'\u2062', u'\u2063', u'\u2064', 
    u'\u206a', u'\u206b', u'\u206c', u'\u206d', u'\u206e', u'\u206f', u'\u3000', u'\u2800',
    u'\u3164', u'\ufeff', u'\uffa0', 
    # u'\u1d159', u'\u1d173', u'\u1d174', u'\u1d175', u'\u1d176', u'\u1d177', u'\u1d178', u'\u1d179', u'\u1d17a', 
]

JOINER_NAMES = ['Virama', 'Invisible_Stacker', 'Brahmi_Joining_Number', 'Joiner', 'Number_Joiner']

@dataclass(unsafe_hash=True)
class Character:
    code: int
    char: str
    category: str = None
    position: str = None
    language: str = None

class IndicCharacters:
    def __init__(self, characters = {}):
        self.characters = characters
        if len(characters) < 1:
            self.init()
        self.positions = set([syl.position for syl in self.characters.values()])
        self.categories = set([syl.category for syl in self.characters.values()])
        self.pairs = Counter(['{}:{}'.format(syl.category, syl.position) for syl in self.characters.values()])

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.characters[index]
        elif isinstance(index, str):
            return self.characters[ord(index)]

    def __contains__(self, index):
        if isinstance(index, int):
            return index in self.characters
        elif isinstance(index, str):
            return ord(index) in self.characters

    def init(self):
        assert(len(self.characters) == 0), len(self.characters)
        for start, end, name in ysd.parse_file_ranges('Blocks.txt'):
            for uni_code in range(start, end+1):
                assert(uni_code not in self.characters)
                self.characters[uni_code] = Character(
                    code=uni_code,
                    char=chr(uni_code),
                    language=name
                )
        for start, end, category in ysd.parse_file_ranges('IndicSyllabicCategory.txt'):
            for uni_code in range(start, end+1):
                assert(uni_code in self.characters and self.characters[uni_code].category == None), self.characters[uni_code]
                self.characters[uni_code].category = category
        for start, end, position in ysd.parse_file_ranges('IndicPositionalCategory.txt'):
            for uni_code in range(start, end+1):
                assert(uni_code in self.characters and self.characters[uni_code].position == None), self.characters[uni_code]
                self.characters[uni_code].position = position

    def keep_by_lang(self, lang):
        if len([language for language in INDIC_LANGS if lang in language]) < 1:
            return None
        return IndicCharacters({key:char for key, char in self.characters.items() if lang in char.language})

    def filter_nonetype_chars(self):
        return IndicCharacters({key:char for key, char in self.characters.items() if (char.position, char.category) != (None, None)})

    def filter_by_category(self, categories):
        return IndicCharacters({key:char for key, char in self.characters.items() if char.category not in categories})

    def to_list(self):
        return [char.char for char in self.characters.values()]

    def to_type2charlist(self):
        output = defaultdict(list)
        for char in self.characters.values():
            if char.position is not None:
                key = char.position
            elif char.category is not None:
                key = char.category
            else:
                continue
            output[key].append(char.char)

            if char.position is None:
                key = "Center"
            else:
                key = 'NonCenter'
            output[key].append(char.char)

        for k in [k for k in output if len(output[k]) < 2]:
            del output[k]
        return dict(output)

class IndicSyllableBreaker:
    def __init__(self):
        self.indic_chars = IndicCharacters()
        self.joiners = sorted([
            character.char
            for _, character in self.indic_chars.characters.items() 
            if character.category in JOINER_NAMES
        ])

    def is_dependent_left(char_obj):
        status = char_obj.position is not None and char_obj.position not in ['Visual_Order_Left']
        status |= char_obj.category in JOINER_NAMES
        status |= char_obj.category in ['Non_Joiner']
        return status

    def is_dependent_right(char_obj):
        status = char_obj.category in JOINER_NAMES
        status |= char_obj.position in ['Visual_Order_Left']
        return status

    def char2obj(self, char):
        char_ord = ord(char)
        if char_ord not in self.indic_chars:
            char_obj = Character(code=char_ord, char=char)
        else:
            char_obj = self.indic_chars[char_ord]
        return char_obj

    def indic_syllabify(self, line):
        syllables, langs, lengths, main_char_ids = [], [], [], []
        right_dependence = False
        for wrd in line.split(' '):
            _syllables, _lengths = [], []
            for char in wrd:
                if char in INVISIBLE_CHARS and len(_syllables) > 0:
                    _syllables[-1] += char # ingoring invisible characters
                    continue

                char_obj = self.char2obj(char)

                if (IndicSyllableBreaker.is_dependent_left(char_obj) or right_dependence) and len(_syllables) > 0 and (len(langs) > 0 and langs[-1] == char_obj.language):
                    assert(len(_syllables) > 0), (char, char_obj, right_dependence, langs[-1], char_obj.language, _syllables, wrd)
                    assert(langs[-1] == char_obj.language), (langs[-1], char_obj.language)
                    _syllables[-1] += char
                    _lengths[-1] += 1 if char_obj.category not in JOINER_NAMES else 0                    

                    # if previouse character is right dependent and this character is right after it.
                    # except if there is no main_char in the syllable where the index will project a diacritic
                    if right_dependence and len(_syllables[-1])-2 == main_char_ids[-1]: # minus 2 because the 'char' has been added into _syllables
                        main_char_ids[-1] += 1

                else:
                    _syllables.append(char)
                    langs.append(char_obj.language)
                    _lengths.append(1)
                    main_char_ids.append(0)

                right_dependence = IndicSyllableBreaker.is_dependent_right(char_obj)
                
            syllables.extend(_syllables)
            lengths.extend(_lengths)
        assert(len(syllables) == len(langs) == len(main_char_ids))
        return syllables, langs, lengths, main_char_ids

    def iter_syllabify(self, line):
        syls, _, _, main_char_ids = self.indic_syllabify(line)
        for syl, main_char_id in zip(syls, main_char_ids):
            yield syl, main_char_id

    def indic_syllabify_line(self, line):
        return ' '.join(self.indic_syllabify(line)[0])

    def char_seg(self, syl):
        return re.sub(r'([{}]) '.format(''.join(self.joiners)), r'\1', ' '.join(syl)).split(' ')