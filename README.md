# Phrase-based-NER-for-HiNER-dataset
Named Entity Recognition using a phrase based approach for the HiNER dataset

This project enhances traditional NER by applying **phrase-level segmentation and feature extraction** before CRF-based tagging.

---

## Dataset: HiNER (Hindi Named Entity Recognition)

The project uses the **HiNER dataset** containing Hindi text tokenized into words, along with corresponding **NER tags** such as:
- `B-PERSON`, `I-LOCATION`, `B-ORGANIZATION`, `O`, etc.

The dataset format:

  {
    "tokens": ["ताजमहल", "भारत", "में", "स्थित", "है"],
    "ner_tags": [4, 4, 22, 22, 22]
  },



## Project Workflow Overview

-  **Data Loading:** Load JSON-structured HiNER dataset.
-  **Train-Test Split:** 80-20 split for training and testing.
-  **Phrase Generation:** Generate all possible token phrases.
-  **Phrase Scoring:** Score each phrase using custom weighted logic.
-  **Phrase Segmentation:** Select best non-overlapping phrases.
-  **Feature Extraction:** Extract morphological & contextual features.
-  **CRF Model Training:** Train CRF model on phrase-level features.
-  **Prediction:** Predict NER tags on unseen sentences.
-  **Evaluation:** Compute accuracy, precision, recall, and F1-score.

---

## Phrase Generation

<code>tokens = ["ताजमहल", "भारत", "में"]
phrases = generate_phrases(tokens)</code>
OUTPUT:
<code>[
  ('ताजमहल', 0, 1),
  ('ताजमहल भारत', 0, 2),
  ('ताजमहल भारत में', 0, 3),
  ('भारत', 1, 2),
  ('भारत में', 1, 3),
  ('में', 2, 3)
]</code>

## Phrase Scoring

Each phrase is scored based on:

Phrase length
Morphological weight (logarithmic scale)
Tag probability (square root scaling)

## Phrase segmentation

Select top-scoring phrases using Beam Search, non-overlapping phrases from the scored list.

<code>Scored Phrases:
  ('ताजमहल भारत', 0, 2) → score: 3.2
  ('भारत', 1, 2)         → score: 2.1 (skipped due to overlap)
  ('में', 2, 3)          → score: 2.5
  
Final Segmentation:
  ('ताजमहल भारत', 0, 2)
  ('में', 2, 3)</code>
  
## Feature Extraction

Each selected phrase generates a feature dictionary including:

- Token prefixes/suffixes
- Token length
- Digit and hyphen checks
- Previous/next token
- Bigram context

Example :
<code>{
  "TOKEN": "ताजमहल",
  "PREFIX_2": "ता",
  "SUFFIX_2": "हल",
  "IS_DIGIT": "0",
  "TOKEN_LEN": "7",
  "PREV_TOKEN": "<START>",
  "NEXT_TOKEN": "भारत",
  "PREV_BIGRAM": "<START>",
  "NEXT_BIGRAM": "ताजमहल भारत"
}</code>

## CRF Model

Model trained using pycrfsuite.Trainer.

Uses extracted features and aligned labels.

Saved as: hindi_ner_phrase_based.crfsuite

---
Reference :
S.-H. Na and Y.-K. Kim, “Phrase-Based Statistical Model for Korean Morpheme Segmentation and POS Tagging,” Journal of KIISE: Software and Applications, vol. 36, no. 5, pp. 325–332, 2009.
