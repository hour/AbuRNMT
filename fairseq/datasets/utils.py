import torch, re, os, glob, shutil
import pycountry
from language_tags import tags

from fairseq import utils
from fairseq.data import indexed_dataset


LANG_TO_SCRIPT = {
    'bg': 'Bengali'
}

# Based on is_master() in
# https://github.com/facebookresearch/fairseq/blob/5307a0e078d7460003a86f4e2246d459d4706a1d/fairseq/distributed/utils.py#L42
def is_master():
    return not torch.cuda.is_available() or torch.cuda.current_device() == 0

def fix_dataset_impl(args):
    data_dirs = utils.split_paths(args.data)
    path = os.path.join(data_dirs[0], "{0}.{1}-{2}.{1}".format('train', args.source_lang, args.target_lang))
    setattr(args, 'dataset_impl', indexed_dataset.infer_dataset_impl(path))

def douplicate_src(args, split, out_prefix):
    replaced_files = []

    for data_dir in utils.split_paths(args.data):
        prefix = os.path.join(data_dir, "{}.{}-{}.".format(split, args.source_lang, args.target_lang))
        out_prefix = os.path.join(data_dir, "{}_{}.{}-{}.".format(out_prefix, split, args.source_lang, args.target_lang))

        for file in glob.glob(r'{}{}.*'.format(prefix, args.source_lang)):
            out_file = file.replace(prefix, out_prefix)
            shutil.copy(file, out_file)
            replaced_files.append((file, out_file))
    return replaced_files

class Lang:
    def __init__(self, name):
        lang = pycountry.languages.lookup(name)
        self.alpha_2 = lang.alpha_2
        self.alpha_3 = lang.alpha_3
        self.name = lang.name
        if lang.alpha_2 in LANG_TO_SCRIPT:
            self.script = LANG_TO_SCRIPT[lang.alpha_2]
        else:
            self.script = pycountry.scripts.lookup(tags.tag(lang.alpha_2).subtags[0].data['record']['Suppress-Script']).name
            self.script = re.sub(r' *\(.*\) *', '', self.script)

