# This expanded provides a comprehensive overview of the repository, including its structure, installation instructions, usage guidelines, and an example workflow.

import editdistance
import pandas as pd
import regex as re
import sys
from config import ConfigManager
from data_utils import clean_text, read_dataset
import glob
import nltk
import nltk.data
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import os
from tqdm import tqdm
import random
import numpy as np

# Add the converter module to the system path
sys.path.insert(0, './converter')
from converter_drinovski import Converter

def get_chars(all_words, all_gs_words):
    """
    Get unique characters from the dataset and create mappings.
    """
    chars = set()

    # Collect all unique characters from the words and ground truth words
    for sentence, gs_sentence in zip(all_words, all_gs_words):
        for w, gs in zip(sentence, gs_sentence):
            chars = chars.union(set(w)).union(set(gs))

    l = sorted(list(chars))

    # Create dictionaries for character to index and index to character mappings
    d = {c: i for i, c in enumerate(l)}
    ind = {i: c for i, c in enumerate(l)}

    return l, d, ind

def create_confusion_matrix(config):
    """
    Create a confusion matrix based on the training and test datasets.
    """
    nltk.download('punkt')
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    # Get all training and test data files
    train_data_files = glob.glob(config.get_train_dataset() + '/*.txt')
    _ = glob.glob(config.get_test_dataset() + '/*.txt')

    # Read datasets
    _, train_aligned_words, train_gs_words, train_labels = read_dataset(train_data_files, sentence_tokenizer, False)
    _, test_aligned_words, test_gs_words, test_labels = read_dataset(train_data_files, sentence_tokenizer, False)

    # Combine training and test data
    all_words = train_aligned_words + test_aligned_words
    all_gs_words = train_gs_words + test_gs_words
    all_labels = train_labels + test_labels

    # Get unique characters and their mappings
    c, d, ind = get_chars(all_words, all_gs_words)
    lenc = len(c)

    # Initialize confusion matrix
    cmatrix = [[0]*lenc for i in range(lenc)]

    # Populate confusion matrix based on the differences between words and ground truth words
    for sentence, gs_sentence, lbls in zip(all_words, all_gs_words, all_labels):
        for w, gs, l in zip(sentence, gs_sentence, lbls):
            if l == 1:
                for o, g in zip(w, gs):
                    if o != g:
                        cmatrix[d[o]][d[g]] += 1

    return cmatrix, d, ind

def get_random_char(word, d):
    """
    Get a random character from the word that exists in the dictionary.
    """
    c = random.choice(word)
    return c if c in d else get_random_char(word, d)

def add_noise(word, cmatrix, d, ind):
    """
    Add noise to a word based on the confusion matrix.
    """
    new_word = word
    c = get_random_char(word, d)

    # Calculate probabilities for character replacement
    total = sum(cmatrix[d[c]])
    if total > 0:
        new_c = [x / total for x in cmatrix[d[c]]]
        randomElement = np.random.choice(range(len(new_c)), p=new_c)
        new_word = new_word.replace(c, ind[randomElement], 1)

    return new_word

def generate_data(converter, config):
    """
    Generate synthetic data by adding noise to the converted text.
    """
    noise_prob = 0.15  # noise 15% of tokens
    cmatrix, d, ind = create_confusion_matrix(config)

    # Get all source files for synthetic data generation
    files = glob.glob(config.get_main_path() + "/data/synthetic_source/*.txt")

    for i, file in enumerate(files):
        raw_text = ''
        with open(file, "r", encoding="utf-8-sig") as f:
            raw_text = f.read()

        sentences = sent_tokenize(raw_text)
        new_sentences = []
        converted_sentences = []
        total_words = 0

        # Convert each sentence using the converter
        for sentence in tqdm(sentences):
            sentence = sentence.lower()
            csentence = converter.convert_text(sentence)
            total_words += len(csentence)

            if len(csentence) > 0:           
                converted_sentences.append(' '.join(csentence))
        
        # Add noise to the converted sentences
        for sentence in converted_sentences:
            edited_sentence = sentence
            words = word_tokenize(sentence)

            for word in words:
                if word in string.punctuation:
                    continue
                
                if random.uniform(0, 1) <= noise_prob:
                    noised_word = add_noise(word, cmatrix, d, ind)
                    edited_sentence = edited_sentence.replace(word, noised_word, 1)
 
            new_sentences.append(edited_sentence)

        # Write the original and noised sentences to the output file
        f = open(config.get_synthetic_dataset() + "/train_data_new_" + os.path.basename(f.name).replace(' ', '_'), "w", encoding="utf-8-sig")
        for ocr_sentence, edited_sentence in zip(converted_sentences, new_sentences):
            f.write(clean_text(ocr_sentence) + '\n')
            f.write(clean_text(edited_sentence) + '\n')
        f.close()

def main():
    """
    Main function to generate synthetic data.
    """
    config = ConfigManager()
    converter = Converter(config)
    generate_data(converter, config)
    
if __name__ == '__main__':
    main()