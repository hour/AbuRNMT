import math
from dataclasses import dataclass, field

import torch
import logging
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion
)

JS_TYPES = ['cipher_daug', 'mvsr', 'max']

logger = logging.getLogger(__name__)

@dataclass
class LabelSmoothedCrossEntropyCriterionJSConfig(LabelSmoothedCrossEntropyCriterionConfig):
    js_alpha: float = field(
        default=1,
        metadata={"help": "alpha hyperparameter for JS loss for CipherDAug"},
    )
    js_warmup: int = field(
        default=1,
        metadata={"help": "WarmUp model with regular x-ent for this many updates before computing JS loss"},
    )
    js_temp: int = field(
        default=1,
        metadata={"help": "A softmax temperature to flatten the prediction distribution"},
    )
    js_type: str = field(
        default='mvsr',
        metadata={"help": "JS type, i.e., {}.".format(str(JS_TYPES))},
    )
    without_js: bool = field(
        default=False,
        metadata={"help": "Without consistency loss."},
    )

@register_criterion("label_smoothed_cross_entropy_js", dataclass=LabelSmoothedCrossEntropyCriterionJSConfig)
class LabelSmoothedCrossEntropyJSCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        js_alpha=0,
        js_warmup=1,
        js_temp=1,
        js_type='cipher_daug',
        without_js=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.js_alpha = js_alpha
        self.js_warmup = js_warmup
        self.js_temp = js_temp
        self.js_type = js_type
        self.without_js = without_js
        logger.info("Alpha for JS Loss set to {} .".format(js_alpha))
        logger.info("Softmax temperature set to {} .".format(js_temp))
        logger.info("JS Loss will start after {} updates.".format(js_warmup))

    def compute_kl_loss(self, model, net_output, prime_net_output, pad_mask=None, reduce=True):
        lprobs = model.get_normalized_probs((net_output[0]/self.js_temp,), log_probs=True)
        prime_lprobs = model.get_normalized_probs((prime_net_output[0]/self.js_temp,), log_probs=True)

        probs = model.get_normalized_probs(net_output, log_probs=False)
        prime_probs = model.get_normalized_probs(prime_net_output, log_probs=False)

        p_loss = torch.nn.functional.kl_div(lprobs, prime_probs, reduction="none")
        q_loss = torch.nn.functional.kl_div(prime_lprobs, probs, reduction="none")

        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def compute_kl_loss_by_mvsr(self, model, net_output, prime_net_output, pad_mask=None, reduce=True):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        prime_lprobs = model.get_normalized_probs((prime_net_output[0]/self.js_temp,), log_probs=True)

        kl_loss = torch.nn.functional.kl_div(prime_lprobs, probs, reduction="none")

        if pad_mask is not None:
            kl_loss.masked_fill_(pad_mask, 0.0)

        if reduce:
            kl_loss = kl_loss.sum()

        return kl_loss

    def forward(self, model, sample, reduce=True, num_updates=None):

        if "prime" not in sample: # when the sample is the valid set
            return super().forward(model, sample, reduce=reduce)

        without_js = self.without_js or (num_updates is not None and num_updates < self.js_warmup)

        # original outputs
        sample_input = sample["net_input"]
        net_output = model(**sample_input)
        og_loss, og_nll_loss = super().compute_loss(model, net_output, sample, reduce=reduce)

        # prime outputs
        prime_sample = sample["prime"]["net_input"]
        prime_sample_input = {
            "src_tokens": prime_sample["src_tokens"],
            "src_lengths": prime_sample["src_lengths"],
            "prev_output_tokens": sample_input["prev_output_tokens"],
        }
        prime_net_output = model(**prime_sample_input)
        prime_loss, prime_nll_loss = super().compute_loss(model, prime_net_output, sample, reduce=reduce)

        target = model.get_targets(sample, net_output)
        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)

        if self.js_type == 'mvsr':
            js_loss = self.compute_kl_loss_by_mvsr(model, net_output, prime_net_output, pad_mask=pad_mask) if not without_js else torch.zeros(1, device=og_loss.device)
            loss = 0.5*og_loss + 0.5*prime_loss + self.js_alpha * js_loss

        elif self.js_type == 'cipher_daug':
            js_loss = self.compute_kl_loss(model, net_output, prime_net_output, pad_mask=pad_mask) if not without_js else torch.zeros(1, device=og_loss.device)
            loss = og_loss + prime_loss + self.js_alpha * js_loss

        elif self.js_type == 'max':
            js_loss = self.compute_kl_loss_by_mvsr(model, net_output, prime_net_output, pad_mask=pad_mask) if not without_js else torch.zeros(1, device=og_loss.device)
            loss = max(og_loss, prime_loss) + self.js_alpha * js_loss

        else:
            raise ValueError('Unknown {}'.format(self.js_type))

        ntokens = sample["ntokens"]
        nsentences = sample["target"].size(0) * 2
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        sample_size = sample_size * 2

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(og_nll_loss.data) if reduce else og_nll_loss.data,
            "js_loss": utils.item(js_loss.data) if reduce else js_loss.data,
            "prime_nll_loss": utils.item(prime_nll_loss.data) if reduce else prime_nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))

        # don't log for valid
        if sample_size == 2 * ntokens:
            js_loss = utils.item(sum(log.get("js_loss", 0) for log in logging_outputs))
            metrics.log_scalar(
                "js_loss",
                js_loss / sample_size,
                sample_size,
                round=3,
            )

            prime_nll_loss = utils.item(sum(log.get("prime_nll_loss", 0) for log in logging_outputs))
            metrics.log_scalar(
                "prime_nll_loss",
                prime_nll_loss / ntokens / math.log(2),
                ntokens,
                round=3,
            )
