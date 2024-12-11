import sys, os

sys.path.append(os.path.join(sys.path[0], '..', 'perturbation', 'src', 'libs'))
import indic_syllable, perturb_substitute
sys.modules["fairseq.augtrans.indic_syllable"] = indic_syllable
sys.modules["fairseq.augtrans.perturb_substitute"] = perturb_substitute

from .datasets import utils, noises
sys.modules["fairseq.augtrans.utils"] = utils
sys.modules["fairseq.augtrans.noises"] = noises

from .datasets import language_triple
sys.modules["fairseq.augtrans.language_triple_dataset"] = language_triple

from .datasets import tokenizer, preprocess, manager
from .criterions import label_smoothed_cross_entropy_js

sys.modules["fairseq.augtrans.tokenizer"] = tokenizer
sys.modules["fairseq.augtrans.preprocess"] = preprocess
sys.modules["fairseq.augtrans.data_manager"] = manager
sys.modules["fairseq.augtrans.label_smoothed_cross_entropy_js"] = label_smoothed_cross_entropy_js

from .tasks.augmented_training import AugmentedTranslation
