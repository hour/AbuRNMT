import codecs, re, json
from argparse import ArgumentParser
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import pairwise_distances

from multiprocessing import Pool

@dataclass
class TextObject:
    text: str
    frequency: int
    main_char_id: int = 0

    def pretty(self):
        return '{}:{}'.format(self.text, self.frequency)

    def __len__(self):
        return len(self.text)
    
    def get_diacritics(self):
        return Counter(list(self.text[:self.main_char_id] + self.text[self.main_char_id+1:]))

###################### functions 
def readbuffer(iterlines, buffer_size=10000):
    lines = []
    for line in iterlines:
        lines.append(line)

        if len(lines) >= buffer_size:
            yield lines
            lines = []

    if len(lines) > 0:
        yield lines

def get_textobjs(lines):
    textobjs, diacritics = [], set()
    for line in lines:
        line, main_char_id, freq = re.split(r'[ \t]', line.rstrip())
        textobj = TextObject(line, frequency=int(freq), main_char_id=int(main_char_id))
        textobjs.append(textobj)
        diacritics.update(set(textobj.get_diacritics()))
    return textobjs, list(diacritics)

def textobjs_to_vectors(textobjs, vocabs, with_tqdm=True):
    tqdm_textobjs = tqdm(textobjs) if with_tqdm else textobjs

    textobj_vecs = np.zeros((len(textobjs), len(vocabs)), dtype=np.float32)
    for row, textobj in enumerate(tqdm_textobjs):
        for diacritic, freq in textobj.get_diacritics().most_common():
            textobj_vecs[row, vocabs[diacritic]] = freq
    return textobj_vecs

def get_textobjs_wrapper(lines, with_tqdm=True):
    textobjs, vocabs = get_textobjs(lines)
    vocabs = {vocab:i for i, vocab in enumerate(vocabs)}
    textobj_vecs = textobjs_to_vectors(textobjs, vocabs, with_tqdm=with_tqdm)
    return textobjs, textobj_vecs

def multiproc_get_textobjs(lines, num_workers):
    textobjs, vocabs = get_textobjs(lines)
    vocabs = {vocab:i for i, vocab in enumerate(vocabs)}

    pool = Pool(processes=num_workers)
    results = []

    for subtextobjs in readbuffer(textobjs):
        results.append(
            pool.apply_async(
                textobjs_to_vectors, (subtextobjs, vocabs, False)
            )
        )

    pool.close()
    pool.join()

    textobj_vecs = []
    for result in results:
        textobj_vecs.extend(result.get())

    return textobjs, np.stack(textobj_vecs)

def add_arguments(parser):
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--pca-dim', type=float)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--alpha', type=float, default=1, help='flatten frequency')

def main():
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    with codecs.open(args.input, 'r', 'utf-8') as fin:
        print('Load text object')
        if args.num_workers > 1:
            textobjs, textobj_vecs = multiproc_get_textobjs(fin, num_workers=args.num_workers)

        else:
            textobjs, textobj_vecs = get_textobjs_wrapper(fin)

    # textobj_vecs = textobj_vecs[1:] - textobj_vecs[0] # not neccessary
    textobj_vecs = textobj_vecs[1:] ** args.alpha
    main_char = textobjs[0]
    textobjs = textobjs[1:]
    # faiss.normalize_L2(textobj_vecs)

    # if args.pca_dim is not None:
    #     print("PCA dimensional reduction {} -> {}".format(len(textobj_vecs[0]), args.pca_dim))
    #     pca_src = len(textobj_vecs[0])
    #     pca_trg = int(args.pca_dim) if args.pca_dim >= 1 else min(len(textobj_vecs[0]), len(textobj_vecs))
    #     pca = faiss.PCAMatrix(pca_src, pca_trg)
    #     pca.train(textobj_vecs)
    #     # textobj_vecs = pca.apply(textobj_vecs)

    #     pca_optimal_dim = pca_trg
    #     if args.pca_dim < 1:
    #         eigenvalues = faiss.vector_to_array(pca.eigenvalues)
    #         for pca_optimal_dim in range(pca_trg-1, 6, -1):
    #             ratio = sum(eigenvalues[:pca_optimal_dim])/sum(eigenvalues[:pca_trg])
    #             if ratio <= args.pca_dim:
    #                 break
    #         print("PCA optimal dimension {}".format(pca_optimal_dim))

    #     A = faiss.vector_to_array(pca.A).reshape((pca_trg, pca_src))[:pca_optimal_dim]
    #     b = faiss.vector_to_array(pca.b)[:pca_optimal_dim]
    #     textobj_vecs = np.matmul(textobj_vecs, A.T) + b
    #     faiss.normalize_L2(textobj_vecs)

    #     assert(pca.is_trained and len(textobj_vecs[0]) == pca_optimal_dim)

    np.savez(
        args.output,
        vocabs=np.array([textobj.text for textobj in textobjs]),
        main_char_ids=np.array([textobj.main_char_id for textobj in textobjs]),
        main_char=main_char.text,
        frequencies=np.array([textobj.frequency for textobj in textobjs]),
        similarity=1/(pairwise_distances(textobj_vecs)+1),
        configs=json.dumps({'alpha': args.alpha}),
    )

if __name__ == '__main__':
    main()
