import numpy as np
from numpy import random
from typing import List

class SubstituteBase:
    def __init__(self, dictionary: List[str], probability: float = 0.6, max_token: float= None, noise_format="{}") -> None:
        self.probability = probability
        self.max_token = max_token
        self.noise_format = noise_format
        self.dictionary = dictionary

    def perturb_word(self, word: str, **kwargs):
        raise NotImplementedError()

    def perturb_line(self, line, **kwargs):
        output, ns_count, total_count = [], 0, 0
        for word in line.split(' '):
            if len(word) < 1: # in case of multiple space
                output.append(word)
                continue
            ns_word, _ns_count, _total_count = self.perturb_word(word, **kwargs)
            output.append(ns_word)
            ns_count += _ns_count
            total_count += _total_count
        return ' '.join(output), ns_count, total_count

    def __call__(self, line):
        return self.perturb_line(line)

class SubHomoGlyph(SubstituteBase):
    def __init__(self, databin_path: str, alpha=0.6, tok_func=lambda x: list(zip(x, [0]*len(x))), skip_func=None, ignored_tgt_tokens=[], **args):
        self.tok_func = tok_func
        self.skip_func = skip_func
        self.alpha = alpha
        data = np.load(databin_path)
        
        self.similarity = data['similarity'].astype('float64')
        # print(self.similarity.shape, file=sys.stderr)

        super().__init__(data['vocabs'], **args)

        self.s2i = {tok:i for i, tok in enumerate(self.dictionary)}
        self.main_char_ids = data['main_char_ids']
        self.main_char = data['main_char'].item()

        ignored_tgt_tokens = [self.s2i[tok] for tok in ignored_tgt_tokens if tok in self.s2i]
        self.similarity[:, ignored_tgt_tokens] = 0
        np.fill_diagonal(self.similarity, 1)

        self.normlize_similiary()

    def set_seed(self, n):
        np.random.seed(n)

    def normlize_similiary(self):
        # self.similarity = np.maximum(self.similarity - self.alpha, 0)
        self.similarity[self.similarity < self.alpha] = 0
        np.fill_diagonal(self.similarity, 0)
        mask = np.count_nonzero(self.similarity, axis=1) > 0
        self.similarity[mask] /= self.similarity[mask].sum(axis=1)[:, None] # normalization
        self.similarity[mask] *= self.probability

        np.fill_diagonal(self.similarity, 1)
        dia_mask = np.eye(self.similarity.shape[0], dtype=bool)
        dia_mask[mask == 0] = 0
        self.similarity[dia_mask] = 1 - self.probability

    def perturb_word(self, word, counter, **kwargs):
        tokens = list(self.tok_func(word))
        
        if self.max_token is None:
            max_token = len(tokens)
        elif self.max_token > 1:
            max_token = self.max_token - counter['ns_count']
        else:
            # max_token = int(len(tokens) * self.max_token)
            max_token = int(self.max_token * counter['total_count'] - counter['ns_count'])
        
        output, ns_count = [], 0
        for tok, main_char_id in tokens:
            org_char = None

            if (self.skip_func is not None and self.skip_func(tok)) or ns_count >= max_token:
                output.append(tok)
                continue

            if tok not in self.dictionary:
                org_char = tok[main_char_id]
                tok = tok[:main_char_id] + self.main_char + tok[main_char_id+1:]

            if tok in self.dictionary:
                ns_tok = random.choice(self.dictionary, 1, p=self.similarity[self.s2i[tok]])[0]
                ns_main_char_id = self.main_char_ids[self.s2i[ns_tok]]
                if org_char is not None:
                    ns_tok = ns_tok[:ns_main_char_id] + org_char + ns_tok[ns_main_char_id+1:]
                    tok = tok[:main_char_id] + org_char + tok[main_char_id+1:]
                ns_count += 1 if tok != ns_tok else 0
                tok = self.noise_format.format(ns_tok) if tok != ns_tok else tok
            elif org_char is not None:
                tok = tok[:main_char_id] + org_char + tok[main_char_id+1:]
            output.append(tok)
        return ''.join(output), ns_count, len(tokens)

    def perturb_line(self, line, **kwargs):
        counter = {
            'total_count': sum([len(list(self.tok_func(word))) for word in line.split(' ')]),
            'ns_count': 0,
        }
        return super().perturb_line(line, counter=counter, **kwargs)