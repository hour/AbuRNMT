from argparse import Namespace
from optparse import OptionError
import sentencepiece as sp
from fairseq.augtrans.noises import (
    delete, insert, substitute, swap,
)
from fairseq.augtrans.indic_syllable import IndicCharacters, IndicSyllableBreaker
from fairseq.augtrans.utils import Lang
import numpy as np
from numpy import random
import re

SENTENCEPIECE = 'sentencepiece'
PERMUTE_SUBWORD_SP = 'permute_subword_sp'
PERMUTE_NGRAM_CHAR_SP = 'permute_ngram_char_sp'
RANDOM_PERMUTE_CHAR_SP = 'karpukhin_permute_char_sp'
PERMUTE_INDIC_SYL_SP = 'permute_indic_syl_sp'
PERMUTE_INDIC_SYL_CHAR_SP = 'ling_permute_indic_syl_sp'
SIMILARITY_AWARE_SAMPLE_NOISES_SP = 'similarity_aware_ns'
TOKENIZER_TYPES = [
    SENTENCEPIECE, PERMUTE_SUBWORD_SP, PERMUTE_NGRAM_CHAR_SP, 
    RANDOM_PERMUTE_CHAR_SP, PERMUTE_INDIC_SYL_SP, PERMUTE_INDIC_SYL_CHAR_SP, 
    SIMILARITY_AWARE_SAMPLE_NOISES_SP
]

OPERSTORS = {
    'delete':delete,
    'insert':insert,
    'substitute':substitute,
    'swap':swap,
}

class SamplingTokenizer:
    def __init__(self, args: Namespace, type: str, path: str):
        if type == SENTENCEPIECE:
            assert(not args.sentencepiece_disable_sampling)
            self.tokenizer = SentencePiece(path, alpha=args.sentencepiece_alpha, nbest_size=args.sentencepiece_nbest)

        elif type == PERMUTE_SUBWORD_SP:
            self.tokenizer = PermuteSubwordSP(
                path,
                operators=list({key.strip() for key in args.src_tokenizer_noises_operators.split(',') if key in OPERSTORS}),
                alpha=args.sentencepiece_alpha, 
                nbest_size=args.sentencepiece_nbest,
                noise_prob=args.noise_prob)

        elif type == PERMUTE_NGRAM_CHAR_SP:
            self.tokenizer = PermuteNgramCharacterSP(
                path,
                operators=list({OPERSTORS[key.strip()] for key in args.src_tokenizer_noises_operators.split(',') if key in OPERSTORS}),
                ngram=args.src_tokenizer_noises_ngram,
                noise_prob=args.noise_prob,
                noise_line_prob=args.noise_prob,
                enable_sampling=not args.sentencepiece_disable_sampling,
            )

        elif type == RANDOM_PERMUTE_CHAR_SP:
            if args.src_tokenizer_lang is None:
                raise OptionError('Please specify --src-tokenizer-lang', args.src_tokenizer_lang)

            try:
               lang = Lang(args.src_tokenizer_lang)
            except:
                raise OptionError('--src-tokenizer-lang {} is not in ISO-639'.format(args.src_tokenizer_lang), args.src_tokenizer_lang)

            self.tokenizer = RandomPermuteCharacterSP(
                path,
                operators=list({OPERSTORS[key.strip()] for key in args.src_tokenizer_noises_operators.split(',') if key in OPERSTORS}),
                codes=IndicCharacters().keep_by_lang(lang.script).to_list(),
                noise_prob=args.noise_prob
            )

        elif type == PERMUTE_INDIC_SYL_SP:
            if args.src_tokenizer_lang is None:
                raise OptionError('Please specify --src-tokenizer-lang')

            try:
               lang = Lang(args.src_tokenizer_lang)
            except:
                raise OptionError('--src-tokenizer-lang {} is not in ISO-639'.args.src_tokenizer_lang)

            self.tokenizer = PermuteIndicSyllableSP(
                path,
                operators=list({OPERSTORS[key.strip()] for key in args.src_tokenizer_noises_operators.split(',') if key in OPERSTORS}),
                indic_chars=IndicCharacters().keep_by_lang(lang.script),
                noise_prob=args.noise_prob
            )

        elif type == PERMUTE_INDIC_SYL_CHAR_SP:
            if args.src_tokenizer_lang is None:
                raise OptionError('Please specify --src-tokenizer-lang')

            try:
               lang = Lang(args.src_tokenizer_lang)
            except:
                raise OptionError('--src-tokenizer-lang {} is not in ISO-639'.args.src_tokenizer_lang)

            self.tokenizer = PermuteIndicSyllableCharacterSP(
                path,
                indic_chars=IndicCharacters().keep_by_lang(lang.script),
                noise_prob=args.noise_prob,
            )

        elif type == SIMILARITY_AWARE_SAMPLE_NOISES_SP:
            if args.src_tokenizer_noises_path is None:
                raise OptionError('Please specify --src-tokenizer-noises-path')
            self.tokenizer = SimilarityAwareNoisesSampling(
                path, args.src_tokenizer_noises_path,
                ns_alpha=args.noise_prob,
                ns_beta=args.src_tokenizer_noises_min_score,
                enable_sampling=not args.sentencepiece_disable_sampling,
            )

        else:
            raise ValueError('Unknown {}'.format(type))

    def __call__(self, line: str) -> list:
        if not line.strip():
            return line
        return self.tokenizer.encode(line)

class BaseTokenizer:
    def encode(self, line: str) -> list:
        raise NotImplementedError()

class SentencePiece(BaseTokenizer):
    def __init__(self, path: str, alpha=0.1, nbest_size=-1):
        self.seeds = np.random.get_state()[1]
        self.current_seed_id = 0
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.Load(path)
        self.alpha = alpha
        self.nbest_size = nbest_size

    def next_seed(self):
        sp.set_random_generator_seed(self.seeds[self.current_seed_id].item())
        self.current_seed_id = (self.current_seed_id + 1) % len(self.seeds)

    def encode(self, line: str) -> list:
        self.next_seed()
        return self.tokenizer.encode(line, out_type=str, enable_sampling=True, alpha=self.alpha, nbest_size=self.nbest_size)

class PermuteSubwordSP(SentencePiece):
    def __init__(self, path: str, operators=['delete', 'insert', 'sub', 'swap'], alpha=0.1, nbest_size=-1, noise_prob=1.0):
        super().__init__(path, alpha=alpha, nbest_size=nbest_size)
        self.noise_prob = noise_prob
        self.operators=operators
        self.probs = np.ones(len(operators))
        self.swap_id=operators.index('swap') if 'swap' in operators else None

    def noise_line(self, subwords: list):
        subwords = subwords.copy()
        chars = list(set(''.join(subwords)))
        i = 0
        while i < len(subwords):
            if self.noise_prob**len(subwords[i]) < random.random():
                i+=1; continue
            
            if self.swap_id is not None:
                swap_chance = 1/len(subwords[i+1]) if i+1 < len(subwords) else 0
                self.probs[self.swap_id] = swap_chance

            probs = self.probs / np.sum(self.probs)
            operator = random.choice(self.operators, size=None, p=probs)
            if operator == 'delete':
                del subwords[i]
                continue
            elif operator == 'insert':
                subwords.insert(i, random.choice(chars))
                i+=1
            elif operator == 'sub':
                subwords[i] = random.choice(chars)
            elif operator == 'swap':
                subwords[i], subwords[i+1] = subwords[i+1], subwords[i]
            i+=1
            
        return subwords

    def encode(self, line: str) -> list:
        self.next_seed()
        return self.noise_line(
            self.tokenizer.encode(line, out_type=str, enable_sampling=True, alpha=self.alpha, nbest_size=self.nbest_size)
        )

class PermuteCharacterSP(SentencePiece):
    def __init__(self, path, operators, noise_prob=1.0, enable_sampling=False):
        super().__init__(path)
        self.operators = operators
        self.noise_prob = noise_prob
        self.enable_sampling = enable_sampling

    def noise_word(self, word, codes):
        if len(word) < 1:
            return word
        return self._noise_word(word, codes)

    def noise_line(self, line):
        prefix, suffix = line.split(line.strip())
        line = line.strip()
        if len(line) < 1:
            return prefix + line + suffix
        return prefix + self._noise_line(line, self.noise_word) + suffix

    def next_seed(self):
        np.random.seed(self.seeds[self.current_seed_id].item())
        super().next_seed()

    def encode(self, line: str) -> list:
        self.next_seed()
        return self.tokenizer.encode(
            self.noise_line(line),
            out_type=str,
            enable_sampling=self.enable_sampling,
            alpha=self.alpha, nbest_size=self.nbest_size,
        )

    def _noise_word(self, word, codes):
        raise NotImplementedError()

    def _noise_line(self, line, fn_noise_word):
        raise NotImplementedError()

class RandomPermuteCharacterSP(PermuteCharacterSP):
    def __init__(self, path, operators, codes, noise_prob=1):
        super().__init__(path, operators, noise_prob)
        self.operators = operators + [None]
        self.operators_dist = [noise_prob / len(operators)] * len(operators) + [1-noise_prob]
        self.codes = codes
        assert(len(self.operators) == len(self.operators_dist)), (self.operators, self.operators_dist)

    def _noise_word(self, word, codes):
        word = list(word)
        for _ in range(1):
            if len(word) < 1:
                break
            operator = random.choice(self.operators, size=None, p=self.operators_dist)
            if operator in [insert, substitute]:
                word = operator(word, codes)
            elif operator is not None:
                word = operator(word)
        return ''.join(word)

    def _noise_line(self, line, fn_noise_word):
        for _ in range(10):
            noisy_line = ' '.join([
                fn_noise_word(word, self.codes) for word in line.split(' ')
            ])
            if noisy_line != line:
                break
        # assert(noisy_line != line), (noisy_line, line)
        return noisy_line

class PermuteNgramCharacterSP(PermuteCharacterSP):
    def __init__(self, path, operators, ngram, noise_prob=1, noise_line_prob=1, enable_sampling=False):
        super().__init__(path, operators, noise_prob, enable_sampling)
        self.ngram = ngram
        self.noise_line_prob = noise_line_prob

    def ngrams_ids(self, word):
        word_len = len(word)
        if self.ngram is None:
            return [(0, word_len)]
        return [(i, min(i+self.ngram, word_len)) for i in range(0, word_len, self.ngram)]

    def _noise_line(self, line, fn_noise_word):
        if random.rand() > self.noise_line_prob:
            return line
        codes = list(set(line) - {' '})
        return ' '.join([
            fn_noise_word(word, codes) for word in line.split(' ')
        ])
    
    def _noise_word(self, word, codes):
        word = list(word)
        output = []
        for s, e in self.ngrams_ids(word):
            if random.rand() > self.noise_prob or e-s < 2:
                output.extend(word[s:e])
                continue
            operator = random.choice(self.operators, size=None)
            if operator in [insert, substitute]:
                output.extend(operator(word[s:e], codes))
            elif operator is not None:
                output.extend(operator(word[s:e]))
            else:
                output.extend(word[s:e])
        return ''.join(output)

class PermuteIndicSyllableSP(PermuteCharacterSP):
    def __init__(self, path, operators, indic_chars, noise_prob=1.0):
        to_exclude_categories = ['Number', 'Number_Joiner', 'Brahmi_Joining_Number', 'Invisible_Stacker', 'Non_Joiner', 'Joiner']
        codes = indic_chars.filter_nonetype_chars().filter_by_category(to_exclude_categories).to_list()
        super().__init__(path, operators, codes, noise_prob=noise_prob)
        self.sylbreaker = IndicSyllableBreaker()

    def syllabify(self, line):
        return self.sylbreaker.indic_syllabify_line(line).split(' ')

    def noise_syllable(self, syllable):
        if len(syllable) < 1:
            return syllable
        
        syllable = list(syllable)
        operator = random.choice(self.operators, size=None, p=self.operators_dist)
        if operator in [insert, substitute]:
            syllable = operator(syllable, self.codes)
        elif operator is not None:
            syllable = operator(syllable)
        return ''.join(syllable)

    def noise_word(self, word):
        if len(word) < 1:
            return word

        word = self.sylbreaker.indic_syllabify_line(word).split(' ')
        nsy_word = []
        for syllable in word:
            for _ in range(1):
                syllable = self.noise_syllable(syllable)

            if len(syllable) < 1:
                continue

            nsy_word.append(syllable)
        return ''.join(nsy_word)

class PermuteIndicSyllableCharacterSP(PermuteIndicSyllableSP):
    def __init__(self, path, indic_chars, noise_prob=1.0):
        operators = [self.indic_delete, self.indic_insert, self.indic_substitute, self.indic_swap]
        super().__init__(path, operators, indic_chars, noise_prob=noise_prob)
        self.type2charlist = indic_chars.to_type2charlist()
        self.indic_chars = indic_chars

    def indic_delete(self, syl):
        if len(syl) > 1:
            i = random.randint(1, len(syl))
            if syl[i] in self.indic_chars and self.indic_chars[syl[i]].category == 'Invisible_Stacker':
                del syl[i:i+2]
            elif syl[i-1] in self.indic_chars and self.indic_chars[syl[i-1]].category == 'Invisible_Stacker':
                del syl[i-1:i+1]
            else:
                del syl[i]
        elif len(syl) > 0:
            del syl[0]
        return syl

    def indic_insert(self, syl):
        if len(syl) > 1:
            i = random.randint(0, len(syl))
            if syl[i] in self.indic_chars and self.indic_chars[syl[i]].category == 'Invisible_Stacker':
                i += 2
            elif syl[i-1] in self.indic_chars and self.indic_chars[syl[i-1]].category == 'Invisible_Stacker':
                i += 1
            j = random.randint(0, len(self.type2charlist['NonCenter']))
            syl.insert(i, self.type2charlist['NonCenter'][j])
            return syl

        elif syl[0] in self.type2charlist['Center']:
            j = random.randint(0, len(self.type2charlist['NonCenter']))
            return syl + [self.type2charlist['NonCenter'][j]]

        else:
            return syl

    def indic_substitute(self, syl):
        i = random.randint(0, len(syl))
        for key in self.type2charlist:
            if key in ['NonCenter', 'Center'] or syl[i] not in self.type2charlist[key]:
                continue
            j = random.randint(0, len(self.type2charlist[key]))
            syl[i] = self.type2charlist[key][j]
        return syl

    def indic_swap(self, syl):
        if len(syl) > 2:
            i = random.randint(1, len(syl) - 1)
            len1, len2 = 1, 1            
            try:
                if syl[i] in self.indic_chars and self.indic_chars[syl[i]].category == 'Invisible_Stacker':
                    len1 += len1
                elif syl[i-1] in self.indic_chars and self.indic_chars[syl[i-1]].category == 'Invisible_Stacker':
                    i -= 1
                    len1 += len1

                if syl[i+len1] in self.indic_chars and self.indic_chars[syl[i+len1]].category == 'Invisible_Stacker':
                    len2 += 1
                
                syl = syl[:i] + syl[i+len1:i+len1+len2] + syl[i:i+len1] + syl[i+len1+len2:]
            except:
                assert(i+len1+len2 > len(syl)), "try here is for length error only"
        return syl

class SimilarityAwareNoisesSampling(SentencePiece):
    def __init__(self, sp_path: str, dia_np_path: str, sp_alpha=0.1, sp_nbest_size=-1, ns_alpha=1.0, ns_beta=1.0, enable_sampling=False):
        super().__init__(sp_path, sp_alpha, sp_nbest_size)

        from fairseq.augtrans.perturb_substitute import SubHomoGlyph

        sylbreaker = IndicSyllableBreaker()

        self.viramas = sorted([
            character.char
            for _, character in sylbreaker.indic_chars.characters.items() 
            if character.language in ['Devanagari', 'Bengali'] and character.category in ['Virama']
        ])

        self.noiser = SubHomoGlyph(
            dia_np_path,
            tok_func=sylbreaker.iter_syllabify,
            skip_func=self.skip_token,
            alpha=ns_beta,
            probability=ns_alpha,
        )

        self.enable_sampling = enable_sampling

    def skip_token(self, tok):
        return re.match(r'.*[{}].*'.format(''.join(self.viramas)), tok) is not None

    def next_seed(self):
        np.random.seed(self.seeds[self.current_seed_id].item())
        self.noiser.set_seed(self.seeds[self.current_seed_id].item())
        super().next_seed()

    def encode(self, line: str) -> list:
        self.next_seed()
        nsy_line, _, _ = self.noiser(line)
        return self.tokenizer.encode(
            nsy_line, out_type=str, enable_sampling=self.enable_sampling,
            alpha=self.alpha, nbest_size=self.nbest_size,
        )
