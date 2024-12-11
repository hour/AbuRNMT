from argparse import Namespace
import typing as tp
import shutil
import logging
import os
from time import sleep

from fairseq import utils
from fairseq.data import Dictionary
from fairseq_cli import preprocess
from fairseq.tokenizer import tokenize_line
from fairseq.tasks.translation import TranslationTask
from fairseq.binarizer import (
    FileBinarizer,
    VocabularyDatasetBinarizer
)
from fairseq.data import indexed_dataset

FAIRSEQ_DICT_FORMAT = 'dict.{lang}.txt'

logger = logging.getLogger('augmented-translation')

def make_dataset(
    vocab: Dictionary,
    input_prefix: str,
    output_prefix: str,
    lang: tp.Optional[str],
    args: Namespace,
    num_workers: int,
    tokenize=tokenize_line,
):
    if args.dataset_impl == "raw":
        # Copy original text file to destination folder
        input_text_file = "{}{}".format(input_prefix, ("." + lang) if lang is not None else "")
        output_text_file = output_prefix + ".{}-{}".format(args.source_lang, args.target_lang)
        output_text_file = preprocess._file_name(output_text_file, lang)

        shutil.copyfile(input_text_file, output_text_file)

    else:
        binarizer = VocabularyDatasetBinarizer(
            vocab,
            append_eos=True,
            tokenize=tokenize,
        )

        input_file = "{}{}".format(input_prefix, ("." + lang) if lang is not None else "")
        output_file = output_prefix + ".{}-{}".format(args.source_lang, args.target_lang)
        output_file = preprocess._file_name(output_file, lang)

        final_summary = FileBinarizer.multiprocess_dataset(
            input_file,
            args.dataset_impl,
            binarizer,
            output_file,
            vocab_size=len(vocab),
            num_workers=num_workers,
        )

        logger.info(f"[{lang}] {input_file}: {final_summary} (by {vocab.unk_word})")

def _preprocess(args, data_path, tokenize_src=tokenize_line, tokenize_tgt=tokenize_line):
    # Use same dictionaries
    first_data_dir = utils.split_paths(args.data)[0]
    src_dict = os.path.join(first_data_dir, FAIRSEQ_DICT_FORMAT.format(lang=args.source_lang))

    src_dict = TranslationTask.load_dictionary(src_dict)
    split = 'train'

    make_dataset(
        src_dict, args.unprocessed_data, os.path.join(data_path, split), 
        args.source_lang, args, args.src_tokenizer_num_workers, tokenize=tokenize_src,
    )

    logger.info('Wrote train to {}'.format(data_path))

def preprocess_iter_src_dataset(args, tokenizer, start=0, size=1, split='train'):
    for i in range(start, start+size):
        for datadir in utils.split_paths(args.data):
            sleep(10)
            dictionary = TranslationTask.load_dictionary(
                os.path.join(datadir, FAIRSEQ_DICT_FORMAT.format(lang=args.source_lang))
            )

            make_dataset(
                dictionary, args.unprocessed_data, os.path.join(datadir, '{}_itr{}'.format(split, i)),
                args.source_lang, args, args.src_tokenizer_num_workers, tokenize=tokenizer,
            )

def is_iter_src_dataset_exist(args, split, index):
    for datadir in utils.split_paths(args.data):
        output_text_file = os.path.join(datadir, '{}_itr{}'.format(split, index))
        output_text_file += ".{}-{}".format(args.source_lang, args.target_lang)
        output_text_file = preprocess._file_name(output_text_file, args.source_lang)
        if not indexed_dataset.dataset_exists(output_text_file, args.dataset_impl):
            return False
    return True
