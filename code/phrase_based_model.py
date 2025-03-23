import pycrfsuite
from sklearn.model_selection import train_test_split
import json
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# hiner tag map
tag_mapping = {
    0: "B-FESTIVAL", 1: "B-GAME", 2: "B-LANGUAGE", 3: "B-LITERATURE", 4: "B-LOCATION", 
    5: "B-MISC", 6: "B-NUMEX", 7: "B-ORGANIZATION", 8: "B-PERSON", 9: "B-RELIGION", 
    10: "B-TIMEX", 11: "I-FESTIVAL", 12: "I-GAME", 13: "I-LANGUAGE", 14: "I-LITERATURE", 
    15: "I-LOCATION", 16: "I-MISC", 17: "I-NUMEX", 18: "I-ORGANIZATION", 19: "I-PERSON", 
    20: "I-RELIGION", 21: "I-TIMEX", 22: "O"
}

# phrase based features
"""
check affixes
digits
token length
hyphens
prev and next tokebn
bigrams
"""
def extract_phrase_features(tokens):
    features = []
    for i in range(len(tokens)):
        token = tokens[i]
        feature = {
            "TOKEN": str(token),
            "PREFIX_1": str(token[:1]),
            "PREFIX_2": str(token[:2]),
            "SUFFIX_1": str(token[-1:]),
            "SUFFIX_2": str(token[-2:]),
            "IS_DIGIT": str(int(token.isdigit())),
            "TOKEN_LEN": str(len(token)),
            "CONTAINS_HYPHEN": str(int("-" in token)),
            "PREV_TOKEN": str(tokens[i-1]) if i > 0 else "<START>",
            "NEXT_TOKEN": str(tokens[i+1]) if i < len(tokens)-1 else "<END>",
            "PREV_BIGRAM": str(tokens[i-1] + ' ' + token) if i > 0 else "<START>",
            "NEXT_BIGRAM": str(token + ' ' + tokens[i+1]) if i < len(tokens)-1 else "<END>",
        }
        features.append(feature)
    return features

# phrase-based segmentations
def generate_phrases(tokens, max_len=3):
    phrases = []
    for i in range(len(tokens)):
        for j in range(i+1, min(i+max_len, len(tokens))+1):
            phrase = ' '.join(tokens[i:j])
            phrases.append((phrase, i, j))
    return phrases


# score func for phrase based segmentation
def score_segmentation(phrases, weights):
    score = 0.0
    for phrase, start, end in phrases:
        phrase_len = end - start
        score += weights.get("length", -1.0) * phrase_len
        score += weights.get("morph_prob", -0.5) * math.log(1 + phrase_len)
        score += weights.get("tag_prob", -0.3) * (phrase_len ** 0.5)
    return score

# optimized- beam search for bestr segment selection
def select_best_segmentation(phrases, weights, beam_width=5):
    candidates = []
    used_indices = set()
    
    for phrase_tuple in phrases:
        if len(phrase_tuple) == 3: 
            phrase, start, end = phrase_tuple
            score = score_segmentation([phrase_tuple], weights)
            candidates.append((phrase, start, end, score))

    candidates.sort(key=lambda x: x[3], reverse=True)
    selected_phrases = []
    
    for phrase, start, end, score in candidates:
        if start not in used_indices:
            selected_phrases.append((phrase, start, end))
            used_indices.update(range(start, end))
    
    return selected_phrases


# load data
def load_hiner_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(entry['tokens'], entry['ner_tags']) for entry in data]

json_file_path = r"HiNER-main\data\original\train.json"
hiner_data = load_hiner_json(json_file_path)


train_set, test_set = train_test_split(hiner_data, test_size=0.2, random_state=42)


# CRF model with phrase based features
trainer = pycrfsuite.Trainer()
trainer.set_params({
    "max_iterations": 200,
    "feature.possible_transitions": True,
    "feature.possible_states": True
})

weights = {"length": -1.0, "morph_prob": -0.5, "tag_prob": -0.3}

for tokens, labels in train_set:
    phrases = generate_phrases(tokens)
    best_segmentation = select_best_segmentation(phrases, weights, beam_width=3)
    features = extract_phrase_features([phrase[0] for phrase in best_segmentation])
    phrase_labels = [labels[min(start, len(labels)-1)] for _, start, _ in best_segmentation]  # Align labels correctly
    if len(features) == len(phrase_labels):  # Ensure lengths match
        trainer.append([list(f.values()) for f in features], list(map(str, phrase_labels)))

# save model
trainer.train("hindi_ner_phrase_based.crfsuite")

# load model
tagger = pycrfsuite.Tagger()
tagger.open("hindi_ner_phrase_based.crfsuite")


def predict(sentence_tokens):
    phrases = generate_phrases(sentence_tokens)
    best_segmentation = select_best_segmentation(phrases, weights, beam_width=3)
    features = extract_phrase_features([phrase[0] for phrase in best_segmentation])
    if len(features) == 0:
        return [(token, "O") for token in sentence_tokens]  # all words included
    predicted_tags = tagger.tag([list(f.values()) for f in features])
    word_to_tag = {phrase[0]: tag_mapping.get(int(tag), "O") for phrase, tag in zip(best_segmentation, predicted_tags)}
    return [(token, word_to_tag.get(token, "O")) for token in sentence_tokens]


# sample
test_sentence = ["देर", "रात", "तक", "जुहू", "चौपाटी", "में", "यह", "नजारा", "आम", "है", "।"]
#["ताजमहल", "भारत", "में", "स्थित", "एक", "भव्य", "स्मारक", "है", "।"]
print(predict(test_sentence))

