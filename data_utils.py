import editdistance

import pandas as pd
import regex as re
import numpy as np

# Define punctuation characters without '@' symbol
PUNCTUATION_WITHOUT_AT = '\ufeff!"#$%&\'()*+, -./:;<=>?[\]^_`{|}~“'

# Define all punctuation characters including '@' symbol
PUNCTUATION_ALL = '!"#$„%&\'()*+, -.@/:;<=>?[\]^_`{|}~“'

pd.options.display.max_rows = 5

class TextProcessor:
    def __init__(self, max_sequence_length, number_of_unique_tokens, char_to_id):
        """
        Initialize the TextProcessor with the given parameters.

        Parameters:
        max_sequence_length (int): The maximum length of sequences.
        number_of_unique_tokens (int): The number of unique tokens.
        char_to_id (dict): A dictionary mapping characters to their corresponding IDs.
        """
        self.max_sequence_length = max_sequence_length
        self.number_of_unique_tokens = number_of_unique_tokens
        self.char_to_id = char_to_id

    def to_ids(self, words):
        """
        Convert a list of words to a 3D numpy array of one-hot encoded character IDs.

        Parameters:
        words (list of str): The list of words to convert.

        Returns:
        numpy.ndarray: A 3D numpy array of shape (len(words), max_sequence_length, number_of_unique_tokens)
                       containing the one-hot encoded character IDs.
        """
        # Initialize the 3D numpy array with zeros
        vector = np.zeros((len(words), self.max_sequence_length, self.number_of_unique_tokens), dtype="float32")

        # Iterate over each word
        for i, word in enumerate(words):
            # Iterate over each character in the word
            for t, char in enumerate(word):
                # Set the corresponding position in the vector to 1.0
                vector[i, t, self.char_to_id[char]] = 1.0
            # Fill the remaining positions in the sequence with the ID for space character
            vector[i, t + 1:, self.char_to_id[' ']] = 1.0

        return vector

def clean_text_all(word):
  word = re.sub(r'\n', '', word)

  return word.lower().strip(PUNCTUATION_ALL).replace('@', '')

def clean_text(word):
  word = re.sub(r'\n', '', word)

  return word.lower().strip(PUNCTUATION_WITHOUT_AT)

def clean_up(word):
  word = re.sub(r'\n', '', word)
  word = word.strip(PUNCTUATION_ALL).lower().replace('@', '')

  return word.strip()

import re

def get_space_positions(sentence, gold_standard_sentence):
    # Strip any leading or trailing whitespace from the gold standard sentence
    gold_standard_sentence = gold_standard_sentence.strip('')

    # Find positions of spaces in the input sentence
    raw_ocr_space_positions = [match.span()[0] for match in re.finditer(" ", sentence)]
    # Find positions of spaces in the gold standard sentence
    gold_standard_ocr_space_positions = [match.span()[0] for match in re.finditer(" ", gold_standard_sentence)]

    # Get the number of spaces in the gold standard sentence
    gold_standard_ocr_space_len = len(gold_standard_ocr_space_positions)

    cursor = 0  # Initialize cursor to track position in gold standard space positions
    usable = []  # List to store usable space positions

    # Iterate over each space position in the input sentence
    for space_position in raw_ocr_space_positions:
        # Move the cursor to the next space position in the gold standard sentence
        while cursor < gold_standard_ocr_space_len and gold_standard_ocr_space_positions[cursor] < space_position:
            cursor += 1

        # If the current space position matches the gold standard space position, add it to usable list
        if cursor < gold_standard_ocr_space_len and gold_standard_ocr_space_positions[cursor] == space_position:
            usable.append(space_position)
    
    # Append the length of the sentence to the usable list
    usable.append(len(sentence))
    
    return usable

def read_data(files, sentence_tokenizer):
    """
    Wrapper function to read dataset using the provided sentence tokenizer.

    Parameters:
    files (list of str): List of file paths to read.
    sentence_tokenizer (Tokenizer): Tokenizer to split sentences.

    Returns:
    tuple: A tuple containing lists of words, aligned words, gold standard words, and labels.
    """
    return read_dataset(files, sentence_tokenizer)

def read_dataset(files, sentence_tokenizer, clean=True):
    """
    Read and process dataset from the given files.

    Parameters:
    files (list of str): List of file paths to read.
    sentence_tokenizer (Tokenizer): Tokenizer to split sentences.
    clean (bool): Flag to indicate whether to clean the text or not.

    Returns:
    tuple: A tuple containing lists of words, aligned words, gold standard words, and labels.
    """
    words = []
    aligned_words = []
    aligned_gs_words = []
    labels = []

    total_files = len(files)
    print("Total files {}".format(total_files))
    
    for file in files:
        print("Processing {}".format(file))
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.readlines()

        # Extract aligned OCR and gold standard sentences from the file
        aligned_ocr = raw_text[1][14:]
        aligned_gs = raw_text[2][14:]

        file_aligned_words = []
        file_words = []
        file_aligned_gs_words = []
        file_labels = []
        
        # Tokenize the aligned OCR sentence into spans
        sentence_spans = sentence_tokenizer.span_tokenize(aligned_ocr)
        
        for sentence_start, sentence_end in sentence_spans:
            aligned_sentence = aligned_ocr[sentence_start:sentence_end]
            gs_sentence = aligned_gs[sentence_start:sentence_end]
            sentence_aligned_words = []
            sentence_aligned_gs_words = []
            sentence_words = []
            sentence_labels = []

            # Get positions of spaces in the aligned sentence
            ocr_space_positions = get_space_positions(aligned_sentence, gs_sentence)
            
            word_start = 0
            for space_position in ocr_space_positions:
                word = aligned_sentence[word_start:space_position]

                if len(word) == 0:
                    word_start = space_position + 1 
                    continue
                
                # Clean the word if the clean flag is set
                cleared_word = clean_text_all(word) if clean else word
                gs_word = gs_sentence[word_start:space_position]
                gs_word = clean_text_all(gs_word) if clean else gs_word

                label = 0
                if cleared_word != gs_word:
                    label = 1
                
                word_start = space_position + 1

                if len(cleared_word) == 0 or len(gs_word) == 0:
                    continue
            
                sentence_labels.append(label)
                sentence_words.append(cleared_word)
                sentence_aligned_words.append(word)
                sentence_aligned_gs_words.append(gs_word) 
            
            if len(sentence_words) > 0 and len(sentence_aligned_gs_words) > 0:
                file_words.append(sentence_words)
                file_aligned_words.append(sentence_aligned_words)
                file_aligned_gs_words.append(sentence_aligned_gs_words)
                file_labels.append(sentence_labels)
        
        words.extend(file_words)
        aligned_words.extend(file_aligned_words)
        aligned_gs_words.extend(file_aligned_gs_words)
        labels.extend(file_labels)

    return words, aligned_words, aligned_gs_words, labels

def read_synthetic_data(files):
    # Initialize lists to store words, aligned words, gold standard words, and labels
    words = []
    aligned_words = []
    aligned_gs_words = []
    labels = []

    # Get the total number of files
    total_files = len(files)
    print("Total files {}".format(total_files))

    # Iterate over each file
    for file in files:
        print("Processing {}".format(file))
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.readlines()

        # Initialize lists to store data for the current file
        file_words = []
        file_aligned_words = []
        file_aligned_gs_words = []
        file_labels = []

        # Iterate over the lines in the file, processing pairs of lines
        for i in range(0, len(raw_text), 2):
            gs_sentence = clean_text(raw_text[i])  # Clean the gold standard sentence
            sentence = clean_text(raw_text[i + 1])  # Clean the OCR sentence
              
            # Initialize lists to store data for the current sentence
            sentence_words = []
            sentence_aligned_words = []
            sentence_aligned_gs_words = []
            sentence_labels = []

            # Get positions of spaces in the OCR sentence
            new_ocr_space_ids = get_space_positions(sentence, gs_sentence)
            
            word_start = 0
            # Iterate over each space position
            for space_id in new_ocr_space_ids:
                word = sentence[word_start:space_id]

                if len(word) == 0:
                    word_start = space_id + 1 
                    continue
                
                # Clean the word and the corresponding gold standard word
                trimmed_word = clean_up(word)
                gs_word = gs_sentence[word_start:space_id]
                gs_word = clean_up(gs_word)

                # Determine the label (1 if words are different, 0 otherwise)
                label = 0
                if trimmed_word != gs_word:
                    label = 1
                
                # Append the data to the respective lists
                sentence_labels.append(label)
                sentence_aligned_words.append(word)
                sentence_words.append(trimmed_word)
                sentence_aligned_gs_words.append(gs_word)
                
                word_start = space_id + 1

            # Append the sentence data to the file lists if they contain any words
            if len(sentence_aligned_words) > 0 and len(sentence_aligned_gs_words) > 0:
                file_aligned_words.append(sentence_aligned_words)
                file_words.append(sentence_words)
                file_aligned_gs_words.append(sentence_aligned_gs_words)
                file_labels.append(sentence_labels)
          
        # Extend the main lists with the data from the current file
        aligned_words.extend(file_aligned_words)
        words.extend(file_words)
        aligned_gs_words.extend(file_aligned_gs_words)
        labels.extend(file_labels)
    
    return aligned_words, words, aligned_gs_words, labels

def levenhstein_distance(sentences):
    raw_ocr_sentence = ''.join(sentences['ocr_sentence'])
    gold_standard_sentence = ''.join(sentences['gs_sentence'])

    return editdistance.distance(raw_ocr_sentence, gold_standard_sentence) / max(len(raw_ocr_sentence), len(gold_standard_sentence))

def filter_data(words_raw, aligned_words, gs_words, labels, max_norm_lev_distance=0.5):
    """
    Filter the data based on the normalized Levenshtein distance between OCR and gold standard sentences.

    Parameters:
    words_raw (list): List of raw words.
    aligned_words (list): List of aligned OCR words.
    gs_words (list): List of gold standard words.
    labels (list): List of labels indicating discrepancies between OCR and gold standard words.
    max_norm_lev_distance (float): Maximum allowed normalized Levenshtein distance to consider a sentence as good.

    Returns:
    tuple: Filtered lists of words, aligned words, gold standard words, and labels.
    """
    # Create a DataFrame to store OCR and gold standard sentences
    sent_stat = pd.DataFrame({
        "ocr_sentence": aligned_words, 
        "gs_sentence": gs_words
    })

    # Calculate the normalized Levenshtein distance for each sentence pair
    sent_stat["sent_levenshtein_distance"] = sent_stat.apply(levenhstein_distance, axis=1)
    
    # Print the number of good sentences and the total number of sentences
    print("good sentences: %s\ntotal sentences: %s" % ((sent_stat["sent_levenshtein_distance"] <= max_norm_lev_distance).sum(), sent_stat.shape[0]))
    
    # Filter the sentences based on the maximum allowed normalized Levenshtein distance
    good_sentences_stat = sent_stat[sent_stat["sent_levenshtein_distance"] <= max_norm_lev_distance]

    # Filter the lists of words, aligned words, gold standard words, and labels based on the good sentences
    words_filtered = np.array(words_raw, dtype=object)[good_sentences_stat.index.tolist()].tolist()
    aligned_words_filtered = np.array(aligned_words, dtype=object)[good_sentences_stat.index.tolist()].tolist()
    gs_words_filtered = np.array(gs_words, dtype=object)[good_sentences_stat.index.tolist()].tolist()
    labels_filtered = np.array(labels, dtype=object)[good_sentences_stat.index.tolist()].tolist()

    return words_filtered, aligned_words_filtered, gs_words_filtered, labels_filtered

def truncate_and_pad(arr, max_sequence_length, tokenizer):   
    return arr[:max_sequence_length] + [tokenizer.pad_token_id] * (max_sequence_length - len(arr))

def tokenize(sentence, text_labels, tokenizer):
    """
    Tokenize a sentence and align the labels with the tokenized words.

    Parameters:
    sentence (list of str): The list of words in the sentence.
    text_labels (list of int): The list of labels corresponding to each word.
    tokenizer (Tokenizer): The tokenizer to use for tokenizing the words.

    Returns:
    tuple: A tuple containing the tokenized sentence and the aligned labels.
    """
    tokenized_sentence = []
    labels = []

    # Iterate over each word and its corresponding label
    for word, label in zip(sentence, text_labels):
        # Tokenize the word into sub-tokens
        tokenized_word = tokenizer.tokenize(word)
        number_of_subtokens = len(tokenized_word)
        
        # Extend the tokenized sentence with the sub-tokens
        tokenized_sentence.extend(tokenized_word)
        
        # Extend the labels with the label repeated for each sub-token
        labels.extend([label] * number_of_subtokens)

    # Add special tokens for the beginning and end of the sentence
    tokenized_sentence = [tokenizer.cls_token] + tokenized_sentence + [tokenizer.sep_token]
    labels = [0] + labels + [0]
    
    return tokenized_sentence, labels

def load_clada(file):
    clada_set = set()
    with open(file, "r", encoding="utf-16") as f:
        raw_text = f.readlines()

    for word in raw_text:
        temp = clean_text(word)

        if temp not in clada_set:
            clada_set.add(temp)

    return clada_set