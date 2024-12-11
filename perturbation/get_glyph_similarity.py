import codecs, re, os
from argparse import ArgumentParser
import numpy as np
import faiss
from tqdm import tqdm

from multiprocessing import Pool
from PIL import ImageFont

from libs.render import TextObject

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

def get_textobjs(lines, font, size):
    textobjs = []
    for line in lines:
        line, main_char_id, freq = re.split(r'[ \t]', line.rstrip())
        textobjs.append(TextObject.from_line(line, freq=int(freq), main_char_id=int(main_char_id), font=font, size=size))

    textobjs, offsets = TextObject.crop_images_in_textobjs(textobjs)

    print('Offsets: {}'.format(offsets))

    return textobjs

def textobjs_to_vectors(textobjs, with_tqdm=True):
    tqdm_textobjs = tqdm(textobjs) if with_tqdm else textobjs

    textobj_vecs = []
    for textobj in tqdm_textobjs:
        vec = textobj.get_glyph_vec(strip=False, binarize=False)
        textobj_vecs.append(vec)

    return textobj_vecs

def get_textobjs_wrapper(lines, font, size, with_tqdm=True):
    textobjs = get_textobjs(lines, font, size)
    textobj_vecs = textobjs_to_vectors(textobjs, with_tqdm=with_tqdm)
    return textobjs, textobj_vecs

def multiproc_get_textobjs(lines, font, size, num_workers):
    textobjs = get_textobjs(lines, font, size)

    pool = Pool(processes=num_workers)
    results = []

    for subtextobjs in readbuffer(textobjs):
        results.append(
            pool.apply_async(
                textobjs_to_vectors, (subtextobjs, False)
            )
        )

    pool.close()
    pool.join()

    textobj_vecs = []
    for result in results:
        textobj_vecs.extend(result.get())

    return textobjs, textobj_vecs

def join_textobj_vecs(textobj_vecs, max_len=None):
    if max_len is None:
        max_len = textobj_vecs[0].shape[0]
    results = np.zeros((len(textobj_vecs),  max_len), dtype=np.float32)
    for i in range(len(textobj_vecs)):
        results[i][:len(textobj_vecs[i])] = textobj_vecs[i]
    return results

def add_arguments(parser):
    parser.add_argument('font_path', type=str)
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--font-size', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--outdir-save-images', type=str)
    parser.add_argument('--pca-dim', type=float)
    parser.add_argument('--cpu', action='store_true')

def main():
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    with codecs.open(args.input, 'r', 'utf-8') as fin:
        print('Load text object')
        if args.num_workers > 1:
            textobjs, textobj_vecs = multiproc_get_textobjs(fin, args.font_path, args.font_size, num_workers=args.num_workers)

        else:
            textobjs, textobj_vecs = get_textobjs_wrapper(fin, args.font_path, args.font_size)

    textobj_vecs = join_textobj_vecs(textobj_vecs)
    textobj_vecs = textobj_vecs[1:] - textobj_vecs[0]
    main_char = textobjs[0]
    textobjs = textobjs[1:]
    faiss.normalize_L2(textobj_vecs)

    if args.pca_dim is not None:
        print("PCA dimensional reduction {} -> {}".format(len(textobj_vecs[0]), args.pca_dim))
        pca_src = len(textobj_vecs[0])
        pca_trg = int(args.pca_dim) if args.pca_dim >= 1 else min(len(textobj_vecs[0]), len(textobj_vecs))
        pca = faiss.PCAMatrix(pca_src, pca_trg)
        pca.train(textobj_vecs)
        # textobj_vecs = pca.apply(textobj_vecs)

        pca_optimal_dim = pca_trg
        if args.pca_dim < 1:
            eigenvalues = faiss.vector_to_array(pca.eigenvalues)
            for pca_optimal_dim in range(pca_trg-1, 6, -1):
                ratio = sum(eigenvalues[:pca_optimal_dim])/sum(eigenvalues[:pca_trg])
                if ratio <= args.pca_dim:
                    break
            print("PCA optimal dimension {}".format(pca_optimal_dim))

        A = faiss.vector_to_array(pca.A).reshape((pca_trg, pca_src))[:pca_optimal_dim]
        b = faiss.vector_to_array(pca.b)[:pca_optimal_dim]
        textobj_vecs = np.matmul(textobj_vecs, A.T) + b
        faiss.normalize_L2(textobj_vecs)

        assert(pca.is_trained and len(textobj_vecs[0]) == pca_optimal_dim)

    font = ImageFont.truetype(f'{args.font_path}.ttf', args.font_size)

    np.savez(
        args.output,
        configs=np.array([args.font_size] + list(font.getname())),
        vocabs=np.array([textobj.text for textobj in textobjs]),
        main_char_ids=np.array([textobj.main_char_id for textobj in textobjs]),
        main_char=main_char.text,
        frequencies=np.array([textobj.frequency for textobj in textobjs]),
        similarity=textobj_vecs.dot(textobj_vecs.T)
    )
    
    # save textobj images
    if args.outdir_save_images is not None:
        os.makedirs(os.path.join(args.outdir_save_images, 'images'), exist_ok=True)
        file_list_path = os.path.join(args.outdir_save_images, 'file_list.txt')
        with codecs.open(file_list_path, 'w', 'utf-8') as fout:
            for textobj in tqdm(sorted(textobjs, key=lambda x:len(x.text), reverse=True)):
                codepoints = '_'.join([hex(ord(char)) for char in textobj.text])
                image_path = './{}.png'.format(os.path.join(args.outdir_save_images, 'images', codepoints))
                # arguments below must be the same as that of in get_textobjs()
                textobj.textimage.get_np(strip=False, binarize=False)[1].save(image_path)
                print('{}\t{}\t{}'.format(textobj.text, codepoints, image_path), file=fout)

if __name__ == '__main__':
    main()
