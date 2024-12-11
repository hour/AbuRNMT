import codecs, json, re
import numpy as np
from argparse import ArgumentParser
from libs.indic_syllable import IndicSyllableBreaker
from libs.perturb_substitute import SubHomoGlyph

def main():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['homoglyph'])
    parser.add_argument('databin')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--args', type=str, default='{}', help='e.g., {"p":1.0, "alpha":1.0}')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--language', type=str, default='Khmer')
    parser.add_argument('--max-token', type=float)
    parser.add_argument('--ignored-tgt-tokens-file', type=str)
    args = parser.parse_args()

    hypars = json.loads(args.args)
    assert(len(set(hypars) - {'p', 'alpha'}) == 0), set(hypars) - {'p', 'alpha'}
    p = float(hypars['p']) if 'p' in hypars else 1.0
    alpha = float(hypars['alpha']) if 'alpha' in hypars else 1.0

    print((hypars, p, alpha))

    fout = codecs.open(args.output, 'w', 'utf-8')

    if args.seed is not None:
        np.random.seed(args.seed)

    noise_format = '<span style="color:red;">{}</span>' if args.debug else '{}'
    line_format = '<p>{}</p>' if args.debug else '{}'
    
    sylbreaker = IndicSyllableBreaker()

    viramas = sorted([
        character.char
        for _, character in sylbreaker.indic_chars.characters.items() 
        if character.language in ['Devanagari', 'Bengali'] and character.category in ['Virama']
    ])
    
    ignored_tgt_tokens = []
    if args.ignored_tgt_tokens_file is not None:
        ignored_tgt_tokens = [
            line.split()[0]
            for line in codecs.open(args.ignored_tgt_tokens_file, 'r', 'utf-8')
        ]

    def skip_token(tok):
        return re.match(r'.*[{}].*'.format(''.join(viramas)), tok) is not None

    def func_tok(line):
        syls, _, _, main_char_ids = sylbreaker.indic_syllabify(line)
        for syl, main_char_id in zip(syls, main_char_ids):
            yield syl, main_char_id

    if args.type == 'homoglyph':
        noiser = SubHomoGlyph(
            args.databin,
            tok_func=func_tok,
            skip_func=skip_token,
            noise_format=noise_format,
            alpha=alpha,
            probability=p,
            max_token=args.max_token,
            ignored_tgt_tokens=ignored_tgt_tokens,
        )

    else:
        raise ValueError('Unknown {}'.format(args.type))

    if args.debug:
        html_head="<!DOCTYPE html><html><head>"
        html_head+="<link href='https://fonts.googleapis.com/css?family=Noto Serif {}' rel='stylesheet'>".format(args.language)
        html_head+="<style>body {{font-family: 'Noto Serif {}';font-size: 22px;}}</style></head><body>".format(args.language)
        print(html_head, end='', file=fout)

    count_ns_list = []
    with codecs.open(args.input, 'r', 'utf-8') as fin:
        for line in fin:
            ns_line, count_ns, count_cc = noiser(line)
            # count_ns_list.append(round(count_ns/count_cc, 1))
            count_ns_list.append(count_ns)
            if args.debug:
                print(line_format.format(line), end='', file=fout)
            print(line_format.format(ns_line), end='', file=fout)

    if args.debug:
        print("</body></html>", end='', file=fout)

    print('Average per-sentence noises: {}'.format(sum(count_ns_list)/len(count_ns_list)))
    num_noisy_line = len(count_ns_list) - count_ns_list.count(0)
    print('Num of noisy lines: {}/{} = {}'.format(num_noisy_line, len(count_ns_list), num_noisy_line/len(count_ns_list)))

    # print(Counter(count_ns_list))
    # count_ns_list = dict(Counter(count_ns_list))
    # for ratio in range(0, 11):
    #     ratio /= 10
    #     print(count_ns_list[ratio] if ratio in count_ns_list else 0)

if __name__ == '__main__':
    main()
