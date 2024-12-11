import os
import itertools
import logging
from fairseq import utils
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.augtrans.language_triple_dataset import LanguageTripleDataset
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)

logger = logging.getLogger('augmented-translation')

class DataManager:
    def __init__(self, cfg, src_dict, tgt_dict, prepend_bos=False, prepend_bos_src=None, append_source_id=False):
        self.cfg = cfg
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.prepend_bos = prepend_bos
        self.prepend_bos_src = prepend_bos_src
        self.append_source_id = append_source_id

    def split_exists(self, split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=self.cfg.dataset_impl)

    def get_prefix(self, split, src, tgt, lang, data_path):
        prefix = None
        if self.split_exists(split, src, tgt, lang, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, tgt))
        elif self.split_exists(split, tgt, src, lang, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split, tgt, src))
        return prefix

    def remove_train_itr_at_epoch(self, epoch):
        import re, os
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        for data_path in paths:
            split = '{}_itr{}'.format('train', epoch)
            src, tgt = self.cfg.source_lang, self.cfg.target_lang
            prefix = self.get_prefix(split, src, tgt, src, data_path)
            for file in os.listdir(data_path):
                file = os.path.join(data_path, file)
                if not re.match(r'{}*'.format(prefix), file):
                    continue
                os.remove(file)
                logger.info('{} is removed.'.format(file))

    def detect_train_itr(self):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        for data_path in paths:
            prefix = self.get_prefix(
                'train_itr1',
                self.cfg.source_lang, 
                self.cfg.target_lang,
                self.cfg.source_lang,
                data_path
            )
            if prefix is not None:
                return True
        return False

    def load_langtriple_dataset(
        self,
        data_path,
        split,
        epoch,
        src,
        tgt,
        combine,
        shuffle=True,
    ):
        src_datasets = []
        prime_src_datasets = []
        tgt_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else "")
            prefix = self.get_prefix(split_k, src, tgt, src, data_path)
            if prefix is None:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, data_path)
                    )
            
            src_prefix = prefix.replace(split_k, '{}_itr{}'.format(split_k, epoch))
            src_dataset = data_utils.load_indexed_dataset(
                src_prefix + src, self.src_dict, self.cfg.dataset_impl
            )
            if self.cfg.truncate_source:
                src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(src_dataset, self.src_dict.eos()),
                        self.cfg.max_source_positions - 1,
                    ),
                    self.src_dict.eos(),
                )
            src_datasets.append(src_dataset)
            
            prime_src_dataset = data_utils.load_indexed_dataset(
                prefix + src, self.src_dict, self.cfg.dataset_impl
            )
            if self.cfg.truncate_source:
                prime_src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(prime_src_dataset, self.src_dict.eos()),
                        self.cfg.max_source_positions - 1,
                    ),
                    self.src_dict.eos(),
                )
            prime_src_datasets.append(prime_src_dataset)

            tgt_dataset = data_utils.load_indexed_dataset(
                prefix + tgt, self.tgt_dict, self.cfg.dataset_impl
            )
            if tgt_dataset is not None:
                tgt_datasets.append(tgt_dataset)

            logger.info(
                "{} {} {}-{} {} examples".format(
                    data_path, split_k, src, tgt, len(src_datasets[-1])
                )
            )

            if not combine:
                break

        assert len(prime_src_datasets) == len(src_datasets) or len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

        if len(src_datasets) == 1:
            src_dataset = src_datasets[0]
            prime_src_dataset = prime_src_datasets[0]
            tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.cfg.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            prime_src_dataset = ConcatDataset(prime_src_datasets, sample_ratios)
            if len(tgt_datasets) > 0:
                tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            else:
                tgt_dataset = None

        if self.prepend_bos:
            assert hasattr(self.src_dict, "bos_index") and hasattr(self.tgt_dict, "bos_index")
            src_dataset = PrependTokenDataset(src_dataset, self.src_dict.bos())
            prime_src_dataset = PrependTokenDataset(prime_src_dataset, self.src_dict.bos())
            if tgt_dataset is not None:
                tgt_dataset = PrependTokenDataset(tgt_dataset, self.tgt_dict.bos())
        elif self.prepend_bos_src is not None:
            logger.info(f"prepending src bos: {self.cfg.prepend_bos_src}")
            src_dataset = PrependTokenDataset(src_dataset, self.cfg.prepend_bos_src)
            prime_src_dataset = PrependTokenDataset(prime_src_dataset, self.cfg.prepend_bos_src)

        eos = None
        if self.append_source_id:
            src_dataset = AppendTokenDataset(
                src_dataset, self.src_dict.index("[{}]".format(src))
            )
            prime_src_dataset = AppendTokenDataset(
                prime_src_dataset, self.src_dict.index("[{}]".format(src))
            )
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(
                    tgt_dataset, self.tgt_dict.index("[{}]".format(tgt))
                )
            eos = self.tgt_dict.index("[{}]".format(tgt))

        align_dataset = None
        if self.cfg.load_alignments:
            align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
            if indexed_dataset.dataset_exists(align_path, impl=self.cfg.dataset_impl):
                align_dataset = data_utils.load_indexed_dataset(
                    align_path, None, self.cfg.dataset_impl
                )

        tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

        return LanguageTripleDataset(
            src_dataset,
            src_dataset.sizes,
            self.src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            self.tgt_dict,
            prime_src=prime_src_dataset,
            prime_src_sizes=prime_src_dataset.sizes,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=shuffle,
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    def load_dataset(self, split, epoch=1, combine=False, dataset_type=LanguagePairDataset):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        if dataset_type == LanguagePairDataset:
            return load_langpair_dataset(
                data_path,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
            )

        elif dataset_type == LanguageTripleDataset:
            return self.load_langtriple_dataset(
                data_path,
                split,
                epoch,
                src,
                tgt,
                combine,
                shuffle=(split != "test"),
            )

        else:
            raise ValueError('Unknown {}'.format(dataset_type))
