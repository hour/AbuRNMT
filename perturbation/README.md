# Glyph Perturbation

Glyph perturbation for Indic languages based on visual similarity.

## Vocabulary construction

```
python get_indic_syllable.py --language Khmer --min-len 1 outputs/km/syllables.txt
```

## Pre-extraction of similarity

Based on Image-based Glyph Embeddings (IGE)

```
python get_glyph_similarity.py <font_path> outputs/km/syllables.txt outputs/km/ige_similarity.npz --font-size 12
```

Based on Diacritic-Count Embeddings (DGE)

```
python get_dcodes_similarity.py <font_path> outputs/km/syllables.txt outputs/km/dge_similarity.npz --alpha 0.1
```

## Perturbation

```
python perturbate.py homoglyph outputs/km/ige_similarity.npz input.txt output.txt --args '{"p":1, "alpha":1}' --language Khmer
```

To output the html style format, specifying --debug. Perturbed glyphs will be highlighted in red.

## Parallelism

Scripts for vocabulary construction and pre-extraction of similarity support parallelism by specifying --num-worker `<num_worker>`.
