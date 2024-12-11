from numpy import random
import codecs
from collections import defaultdict

KHMER_UNICODES=[chr (0x1780+x) for x in range (49)] + [chr (0x17b6+x) for x in range (28)] + [chr (0x17dd)]
OPERSTORS = ['delete', 'insert', 'substitute', 'swap']

############# indices specific random permutation
def delete(word, n_char=1, min_len=2):
    assert(isinstance(word, list)), type(word)
    assert(n_char < len(word)), '{} >= {}'.format(n_char, len(word))
    if len(word) < min_len:
        return word
    ids = random.choice(len(word), size=n_char)
    return [char for i, char in enumerate(word) if i not in ids]

def insert(word, codes, n_char=1, min_len=2):
    assert(isinstance(word, list)), type(word)
    assert(n_char < len(word)), '{} >= {}'.format(n_char, len(word))
    if len(word) < min_len:
        return word
    output = []
    ids = random.choice(len(word), size=n_char)
    for i, char in enumerate(word):
        if i in ids:
            j = random.randint(0, high=len(codes))
            output.append(codes[j])
        output.append(char)
    return output

def substitute(word, codes, n_char=1, min_len=2):
    assert(isinstance(word, list)), type(word)
    assert(n_char < len(word)), '{} >= {}'.format(n_char, len(word))
    if len(word) < min_len:
        return word
    word = word.copy()
    ids = random.choice(len(word), size=n_char)
    for i in ids:
        j = random.randint(0, len(codes))
        word[i] = codes[j]
    return word

def swap(word, n_char=1, min_len=2):
    assert(isinstance(word, list)), type(word)
    assert(n_char < len(word)), '{} >= {}'.format(n_char, len(word))
    if len(word) < min_len:
        return word
    ids = random.choice(len(word)-1, size=n_char)
    for i in ids:
        word[i], word[i+1] = word[i+1], word[i]
    return word

####################### Dictionary-based Noies Sampling 
def load_dict(path, min_score=0.0, max_score=1.0):
    tgts, distribs = defaultdict(list), defaultdict(list)
    with codecs.open(path, 'r', 'utf-8') as fin:
        for line in fin:
            score, src, tgt, _ = line.strip().split('\t', 3)
            score = float(score)
            
            if score < min_score or score > max_score:
                continue

            tgts[src].append(tgt)
            # distribs[src].append(1-score/100)
            distribs[src].append(score)

    for key in tgts:
        tgts[key].append(key)
        distribs[key].append(1.0)
        total = sum(distribs[key])
        distribs[key] = [p/total for p in distribs[key]]

    return tgts, distribs

def sampling_substitute(token, samples, distributions, probability=1.0):
    """
        Random substitute one letter in a word
    """
    if random.rand() > probability:
        return token
    else:
        return random.choice(samples, 1, p=distributions)[0]

def inject_noises(ws, noise_samples, noise_distribs, probability=1.0):
    return [
        sampling_substitute(w, noise_samples[w], noise_distribs[w], probability=probability) if w in noise_samples else w
        for w in ws
    ]

