from time import sleep
from threading import Thread
import logging
import torch

from fairseq.tasks.translation import TranslationTask, register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq import utils
from fairseq.data import LanguagePairDataset

from fairseq.augtrans.language_triple_dataset import LanguageTripleDataset
from fairseq.augtrans.preprocess import preprocess_iter_src_dataset, is_iter_src_dataset_exist
from fairseq.augtrans.tokenizer import SamplingTokenizer, TOKENIZER_TYPES
from fairseq.augtrans.data_manager import DataManager
from fairseq.augtrans.utils import is_master, fix_dataset_impl, douplicate_src
from fairseq.augtrans.label_smoothed_cross_entropy_js import LabelSmoothedCrossEntropyJSCriterion

SLEEP_INTERVAL = 60 * 0.5  # while waiting for data processing to finish

logger = logging.getLogger('augmented-translation')

@register_task('augmented-translation')
class AugmentedTranslation(TranslationTask):
    """Fairseq task for training a translation model while preprocessing the data each epoch.
    The motivation is to train using different augmentation and regularization techniques,
    such as subword sampling and BPE-dropout.
    This is only a template, though. You'll probably need to add your arguments in add_args method,
    and you'll definitely need to write your preprocessing code in _preprocess method."""

    @staticmethod
    def add_args(parser):
        # Add your arguments here
        parser.add_argument('--unprocessed-data', required=True,
                            help='Path of un-processed training data')
        parser.add_argument('--src-tokenizer-num-workers', type=int, default=1)
        parser.add_argument('--src-tokenizer-type', choices=TOKENIZER_TYPES, type=str)
        parser.add_argument('--src-tokenizer-path', type=str)
        parser.add_argument('--src-tokenizer-lang', type=str)
        parser.add_argument('--src-tokenizer-noises-path', type=str)
        parser.add_argument('--src-tokenizer-noises-min-score', type=float, default=0)
        parser.add_argument('--src-tokenizer-noises-max-score', type=float, default=1.0)
        parser.add_argument('--src-tokenizer-noises-ngram', type=int)
        parser.add_argument('--src-tokenizer-noises-operators', type=str, default='delete,insert,substitute,swap')

        parser.add_argument('--sentencepiece-alpha', type=float, default=0.2)
        parser.add_argument('--sentencepiece-nbest', type=int, default=-1)
        parser.add_argument('--sentencepiece-disable-sampling', action='store_true')

        parser.add_argument('--noise-prob', type=float, default=0.4)
        
        TranslationTask.add_args(parser)

    @classmethod
    def setup_task(cls, args, **kwargs):
        if not getattr(args, 'source_lang', None) or not getattr(args, 'target_lang', None):
            raise ValueError('Please specify --source-lang and --target-lang')

        if (not getattr(args, 'src_tokenizer_type', None)) != (not getattr(args, 'src_tokenizer_path', None)):
            raise ValueError('Please specify --src-tokenizer-type and --src-tokenizer-path')

        src_tokenizer = SamplingTokenizer(args, args.src_tokenizer_type, args.src_tokenizer_path)

        fix_dataset_impl(args)
        logger.info('dataset_impl: {}'.format(args.dataset_impl))

        task = super().setup_task(args, **kwargs)
        task.set_src_tokenizer(src_tokenizer)

        if is_master():
            for old, new in douplicate_src(args, split='train', out_prefix='prime'):
                logger.info('copy {} to {}'.format(old, new))

            if task.data_manager.detect_train_itr():
                logger.warn('Deteted train iterative datasets.')
                logger.info('Clearing all iterative datasets')
                for i in range(1, args.max_epoch+1):
                    task.data_manager.remove_train_itr_at_epoch(i)

            preprocess_iter_src_dataset(args, tokenizer=src_tokenizer, start=1, size=1, split='train')
            task.dataset_thread = Thread(
                target=preprocess_iter_src_dataset, 
                args=[args], 
                kwargs={'tokenizer':src_tokenizer, 'start':2, 'size':args.max_epoch-1, 'split':'train'}
            )
            task.dataset_thread.setDaemon(True)
            task.dataset_thread.start()

            import atexit
            def remove_all_itrs():
                logger.info('Clearing all iterative datasets')
                for i in range(1, args.max_epoch+1):
                    task.data_manager.remove_train_itr_at_epoch(i)
            atexit.register(remove_all_itrs)

        return task

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.data_manager = DataManager(self.cfg, src_dict, tgt_dict)
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.dataset_thread = None

    def set_src_tokenizer(self, tokenizer):
        self.src_tokenizer = tokenizer

    def set_tgt_tokenizer(self, tokenizer):
        self.tgt_tokenizer = tokenizer

    def wait_until_file_is_created(self, split, epoch):
        if epoch > self.cfg.max_epoch:
            return

        for i in range(10):
            if not is_iter_src_dataset_exist(self.cfg, split, epoch):
                sleep(SLEEP_INTERVAL)
            else:
                break
        if not is_iter_src_dataset_exist(self.cfg, split, epoch):
            raise ValueError('Please consider longer SLEEP_INTERVAL')

    # return True to re-call load_dataset() at every epoch
    def has_sharded_data(self, split):
        return split == 'train'

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if epoch > self.cfg.max_epoch:
            if split != 'train' or not is_master():
                return
            
            # Here is the end of epoch
            logger.info('Clearing all iterative datasets')
            for i in range(1, self.cfg.max_epoch+1):
                self.data_manager.remove_train_itr_at_epoch(i)
            return

        if split == 'train':
            self.wait_until_file_is_created(split, epoch)
            dataset_type = LanguageTripleDataset
        else:
            dataset_type = LanguagePairDataset

        super().load_dataset(split, epoch, combine, **kwargs)
        self.datasets[split] = self.data_manager.load_dataset(split, epoch=epoch, combine=combine, dataset_type=dataset_type)

    def get_batch_iterator(
        self,
        dataset,
        **kwargs,
    ):
        return super().get_batch_iterator(dataset, **kwargs)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        if not isinstance(criterion, LabelSmoothedCrossEntropyJSCriterion):
            return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad=ignore_grad)

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, num_updates=update_num)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
