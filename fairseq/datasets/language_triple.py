# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import LanguagePairDataset, data_utils
from fairseq.data.language_pair_dataset import collate as language_pair_collate

logger = logging.getLogger(__name__)

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    batch = language_pair_collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        input_feeding=input_feeding,
        pad_to_length=pad_to_length,
        pad_to_multiple=pad_to_multiple,
    )
    
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    assert(torch.eq(src_lengths, batch["net_input"]["src_lengths"]).all())

    # prime source
    prime_src_tokens = merge(
        "prime_source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )

    # sort by descending source length
    prime_src_lengths = torch.LongTensor([s["prime_source"].ne(pad_idx).long().sum() for s in samples])
    prime_src_lengths = prime_src_lengths.index_select(0, sort_order)

    prime_src_tokens = prime_src_tokens.index_select(0, sort_order)

    batch["prime"] = {
        "net_input": {
            "src_tokens": prime_src_tokens,
            "src_lengths": prime_src_lengths,
        }
    }

    if "prev_output_tokens" in batch["net_input"]:
        batch["prime"]["net_input"]["prev_output_tokens"] = batch["net_input"]["prev_output_tokens"]

    return batch

class LanguageTripleDataset(LanguagePairDataset):
    """
    A triple of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        prime_src=None,
        prime_src_sizes=None,
        **kwargs,
    ):
        super().__init__(
            src,
            src_sizes,
            src_dict,
            tgt,
            tgt_sizes,
            tgt_dict,
            **kwargs,
        )

        # triple
        if prime_src is not None:
            assert len(src) == len(prime_src), "Source and Prime Source must contain the same number of examples"
        self.prime_src = prime_src
        self.prime_src_sizes = np.array(prime_src_sizes)
        # TODO: add assertions for unk/eos/pad between all dicts

    def __getitem__(self, index):
        example = super().__getitem__(index)

        prime_src_item = self.prime_src[index]
        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.prime_src[index][0] != bos:
                prime_src_item = torch.cat([torch.LongTensor([bos]), self.prime_src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.prime_src[index][-1] == eos:
                prime_src_item = self.prime_src[index][:-1]

        example["prime_source"] = prime_src_item

        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        # adding switcher, switchout_tau and raml_tau args to collate function for SwithOut
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
        return res

