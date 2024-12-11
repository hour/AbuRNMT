# AbuRNMT

The purpose of this repository is to reproduce the experiments of our EACL paper on [Robust Neural Machine Translation for Abugidas by Glyph Perturbation](https://aclanthology.org/2024.eacl-short.27/).

# Dependency

* Fairseq >= 0.12.2
* Faiss >= 1.7.2
* Pytorch >= 1.13.1
* Numpy >= 1.21.5
* Pillow >= 9.4.0
* other dependencies
```
pip install sentencepiece pycountry language-tags iso639-lang youseedee datasets scikit-learn
```

# Perturbation

The codes are in [here](perturbation)

# Robust Training

The codes are in [here](fairseq)

# Citation

If you use any codes in this repository, please cite our paper as follows:

```
@inproceedings{kaing-etal-2024-robust,
    title = "Robust Neural Machine Translation for Abugidas by Glyph Perturbation",
    author = "Kaing, Hour  and
      Ding, Chenchen  and
      Tanaka, Hideki  and
      Utiyama, Masao",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-short.27",
    pages = "311--318",
}
```

# License

This software is published under the MIT-license.

# Acknowledgement

* Codes in fairseq directory are partially extended from [fairseq](https://github.com/facebookresearch/fairseq) and [cipherdaug-nmt](https://github.com/protonish/cipherdaug-nmt).
* We'd like to thank Raj Dabre for his help to release this repository. 