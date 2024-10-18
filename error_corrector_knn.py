import re

from data_utils import clean_up
from dataset_parser import DatasetParser

class ErrorCorrectorKnn:
    def __init__(self, config):
        # Initialize with configuration
        self.config = config
        # Define punctuation characters to strip from words
        self.punctuation = '!"#$%&\'()*+, ./:;<=>?@[\]^_`{|}~\n'
        # Load the CLADA dictionary from a file and clean up the words
        self.clada_dictionary = set(line.strip(self.punctuation).lower() for line in open(self.config.get_clada_dictionary(), encoding='utf-16', mode="r"))

        # Initialize the dataset parser
        self.dataset_parser = DatasetParser(config)

        # Read training dataset
        train_words, train_aligned_words, train_gs_words, train_labels = self.dataset_parser.read_train_dataset()
        self.freq_dict = {}

        # Build frequency dictionary from training dataset
        for sentence in train_gs_words:
            for word in sentence:
                word = clean_up(word)

                if word not in self.freq_dict:
                    self.freq_dict[word] = 0

                self.freq_dict[word] = self.freq_dict[word] + 1

    def call(self, token):
        # Check if token is in CLADA dictionary or not eligible for correction
        if self.check_clada_dict(token) or self.token_not_eligible(token):
            return token
        
        # Attempt to correct the token
        corrected_token = self.correct(token)

        # If no correction found, return the original token
        if corrected_token is None:
            corrected_token = token

        return corrected_token

    def correct(self, token):
        # Generate candidates with Levenshtein distance 1
        candidates = self.candidates_lev_distance_1(token)

        if len(candidates) > 0:
            # Select the top candidate based on frequency
            top_candidate = sorted(candidates, key=lambda candidate: self.frequency(candidate))[:1]

            return top_candidate[0] if top_candidate is not None and len(top_candidate) > 0 else None
        else:
            # Generate candidates with Levenshtein distance 2
            candidates = self.candidates_lev_distance_2(token)

            if len(candidates) > 0:
                # Select the top candidate based on frequency
                top_candidate = sorted(candidates, key=lambda candidate: self.frequency(candidate))[:1]

                return top_candidate[0] if top_candidate is not None and len(top_candidate) > 0 else None

        return None

    def token_not_eligible(self, token):
        # Check if token contains any Cyrillic characters
        return re.search('[а-яѫѣꙝѧωѝѹ]', token.lower()) is None

    def check_clada_dict(self, token):
        # Check if token is in CLADA dictionary
        return True if token in self.clada_dictionary else False

    def frequency(self, word):
        # Get the frequency of a word from the frequency dictionary
        return self.freq_dict[word] if word in self.freq_dict else 0

    def dist1(self, word):
        # Generate all possible edits with Levenshtein distance 1
        letters = 'абвгдежзийклмнопрстуфхцчшщъьюяѫѣꙝѧωѝѹ'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        
        return set(deletes + transposes + replaces + inserts)

    def dist2(self, word):
        # Generate all possible edits with Levenshtein distance 2
        return (e2 for e1 in list(self.dist1(word)) for e2 in self.candidates_lev_distance_1(e1))

    def known(self, words):
        # Filter words that are known (present in CLADA dictionary)
        return set(w for w in words if w in self.clada_dictionary)

    def candidates_lev_distance_1(self, word):
        # Get known words with Levenshtein distance 1
        return list(self.known(self.dist1(word)))

    def candidates_lev_distance_2(self, word):
        # Get known words with Levenshtein distance 2
        return list(self.known(self.dist2(word)))