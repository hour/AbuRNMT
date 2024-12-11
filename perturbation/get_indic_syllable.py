import codecs
from collections import Counter
from argparse import ArgumentParser
from multiprocessing import Pool
from iso639 import Lang
from datasets import load_dataset
from tqdm import tqdm

from libs.indic_syllable import INDIC_LANGS, IndicSyllableBreaker

def get_syllables(args, lines, sylbreaker, language, show_progress=False):
    results = []
    no_consonant = Counter()
    end_vis_order_left = Counter()

    if show_progress:
        lines = tqdm(lines)

    for line in lines:
        line = line.strip()
	
        for syl, lang, length, main_char_id in zip(*sylbreaker.indic_syllabify(line)):
            if (language is not None and lang != language) or args.min_len > length or length > args.max_len:
                continue

            # illegal syllable if right_dependent character begin the syllable
            if IndicSyllableBreaker.is_dependent_left(sylbreaker.char2obj(syl[0])):
                no_consonant.update({syl:1})
                continue

            # illegal syllable if left_dependent character end the syllable, e.g., Thai
            if sylbreaker.char2obj(syl[-1]).position in ['Visual_Order_Left']:
                end_vis_order_left.update({syl:1})
                continue

            results.append((syl, main_char_id))
    return Counter(results), no_consonant, end_vis_order_left

def readlines_dataset(dataset, buffer_size=100000):
    dataset_len = len(dataset)
    for i in range(0, dataset_len, buffer_size):
        yield dataset[i:min(i+buffer_size, dataset_len)]

def readlines_inputfile(iterlines, buffer_size=100000):
    lines = []
    for line in iterlines['text']:
        lines.append(line)

        if len(lines) >= buffer_size:
            yield {'text':lines}
            lines = []

    if len(lines) > 0:
        yield {'text':lines}

def main():
    parser = ArgumentParser()
    parser.add_argument('--min-len', default=0, type=int)
    parser.add_argument('--max-len', default=1000, type=int)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--language', type=str, choices=INDIC_LANGS)
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('output')

    args = parser.parse_args()

    sylbreaker = IndicSyllableBreaker()

    # websites
    # https://data.statmt.org/cc-100/
    # https://huggingface.co/datasets/cc100

    if args.language == 'Myanmar':
        lang = Lang('Burmese')
    elif args.language == 'Devanagari':
        lang = Lang('Hindi')
    else:
        lang = Lang(args.language)

    name = "cc100"

    if args.input_file is not None:
        dataset = {'text':codecs.open(args.input_file, 'r', 'utf-8')}
        readlines = readlines_inputfile
    else:
        dataset = load_dataset(name, lang=lang.pt1, cache_dir="/share03/mhkaing/.cache", split='train')
        readlines = readlines_dataset

    if args.num_workers < 2:
        results, no_consonant, end_vis_order_left = get_syllables(args, dataset['text'], sylbreaker, args.language, True)

    else:
        pool = Pool(processes=args.num_workers)
        asynce_results = []
        buffer_size=100000
        total = round(dataset.num_rows/buffer_size, 0) if args.input_file is None else None
        for lines in tqdm(readlines(dataset, buffer_size=buffer_size), total=total):
            asynce_results.append(
                pool.apply_async(
                    get_syllables, (args, lines['text'], sylbreaker, args.language, False)
                )
            )
        pool.close()
        pool.join()

        results, no_consonant, end_vis_order_left = Counter(), Counter(), Counter()
        for result in asynce_results:
            _results, _no_consonant, _end_vis_order_left = result.get()
            results += _results
            no_consonant += _no_consonant
            end_vis_order_left += _end_vis_order_left
            # results.extend(result.get())

    visual_order_left = [
        character.char
        for _, character in sylbreaker.indic_chars.characters.items() 
        if character.language == args.language and character.position in ['Visual_Order_Left']
    ]

    consonants = sorted([
        character.char
        for _, character in sylbreaker.indic_chars.characters.items() 
        if character.language == args.language and character.category in ['Consonant', 'Vowel_Independent']
    ])

    non_consonants_center = sorted([
        character.char
        for _, character in sylbreaker.indic_chars.characters.items() 
        if character.language == args.language and character.category not in ['Consonant', 'Vowel_Independent'] and character.position is None
    ])

    joiners = sorted([
        character.char
        for _, character in sylbreaker.indic_chars.characters.items() 
        if character.language == args.language and character.category in ['Virama', 'Invisible_Stacker', 'Brahmi_Joining_Number', 'Joiner', 'Number_Joiner']
    ])

    newresults = Counter()
    for character in consonants + non_consonants_center:
        newresults.update({('g_2|'+character, 0):1}) # single consonant/non_consonant_center

    for (syl, main_char_id), freq in results.most_common():
        if syl[-1] in joiners:
            syl = syl[:-1]

        if len(syl) > 1:
            offset = 0
            for i in range(len(syl)):
                if syl[i] in visual_order_left:
                    offset = i + 1
                    continue
                break

            if offset < len(syl) and syl[offset] in consonants:
                syl = syl[:offset] + consonants[0] + syl[offset+1:]
                syl = "g_1|" + syl # well form glyphs
            else:
                syl = "g_3|" + syl # center character is not consonant

            newresults.update({(syl, main_char_id):freq})

        else:
            assert(syl in consonants + non_consonants_center), syl

    for character, freq in no_consonant.most_common():
        newresults.update({('g_4|'+character, 0):freq}) # illegal zero consonant
    
    for character, freq in end_vis_order_left.most_common():
        newresults.update({('g_5|'+character, 0):freq}) # illegal end visual order left

    with codecs.open(args.output, 'w', 'utf-8') as fout:
        print('{}\t0\t1'.format(consonants[0]), file=fout)
        for (syl, main_char_id), frq in sorted(newresults.most_common(), key=lambda x:(x[0][0].split('|')[0], x[1]), reverse=True):
            if args.verbose:
                print('{}\t{}\t{}'.format(syl, main_char_id, frq), file=fout)
                continue

            group, syl = syl.split('|')
            if group != 'g_1':
                continue
            print('{}\t{}\t{}'.format(syl, main_char_id, frq), file=fout)

if __name__ == '__main__':
    main()
