from error_detection_models import CladaErrorDetectionModel, FastTextErrorDetectionModel
from data_utils import PUNCTUATION_ALL, clean_text
from config import ConfigManager
from dataset_parser import DatasetParser
from error_corrector_knn import ErrorCorrectorKnn

from Levenshtein import distance as levenshtein_distance

def evaluate_knn(config, parser):
    # Initialize error detection and correction models
    clada = CladaErrorDetectionModel(config)
    knn = ErrorCorrectorKnn(config)

    # Read test dataset
    test_words, test_aligned_words, test_gs_words, test_labels = parser.read_test_dataset()

    distances0 = []  # List to store Levenshtein distances between OCR and GS sentences
    distances1 = []  # List to store Levenshtein distances between corrected and GS sentences

    len_cor = []  # List to store lengths of corrected words
    cnt = 0  # Counter for processed sentences
    n = len(test_aligned_words)  # Total number of sentences

    # Iterate over each sentence in the test dataset
    for sentence, ocr_sentence, gs_sentence, labels in zip(test_words, test_aligned_words, test_gs_words, test_labels):
        gs_sentence_string = ' '.join(gs_sentence)  # Join GS sentence words into a single string
        ocr_sentence_string = ' '.join(sentence)  # Join OCR sentence words into a single string
        corrected = []  # List to store corrected words
        raw = []  # List to store original words
        cnt += 1  # Increment counter

        print("{}/{}".format(cnt, n))  # Print progress

        # Iterate over each word in the sentence
        for word, label, gs in zip(sentence, labels, gs_sentence):
            corrected_word = word  # Initialize corrected word as the original word
            x = None
            if label == 1:
                x = knn.call(word)  # Call KNN model if the word is labeled as erroneous
            # print(x)
            if x is not None:
                corrected_word = str(x)  # Update corrected word if KNN model returns a correction
                len_cor.append(len(gs))  # Append length of GS word to len_cor
            else: 
                corrected_word = word  # Keep the original word if no correction is made

            raw.append(word)  # Append original word to raw list
            corrected.append(corrected_word)  # Append corrected word to corrected list

        ocr_sentence_string = ' '.join(raw)  # Join raw words into a single string
        corrected_sentence_string = ' '.join(corrected)  # Join corrected words into a single string

        # Calculate Levenshtein distances
        levdist0 = levenshtein_distance(ocr_sentence_string, gs_sentence_string)  # Distance between OCR and GS
        levdist1 = levenshtein_distance(corrected_sentence_string, gs_sentence_string)  # Distance between corrected and GS

        distances0.append(levdist0)  # Append distance to distances0
        distances1.append(levdist1)  # Append distance to distances1

    # Calculate improvement in Levenshtein distance
    improvement = (sum(distances0) - sum(distances1)) / (sum(len_cor) + 1)
    print("The improvement is {:.3f}%".format(improvement * 100))  # Print improvement percentage

def extend_dictionary(gs_words, clada):
    all_words = []
    for word in gs_words:
        word = [clean_text(w) for w in word]  # Clean each word in the GS words
        word = list(filter(lambda w: clada.forward(w) and len(w) > 2, word))  # Filter words using Clada model and length

        all_words.extend(word)  # Extend all_words list with filtered words
          
    return all_words  # Return the extended dictionary

def main():
    config = ConfigManager()  # Initialize configuration manager
    datasetparser = DatasetParser(config)  # Initialize dataset parser with config
    evaluate_knn(config, datasetparser)  # Evaluate KNN model
    
if __name__ == '__main__':
    main()  # Run main function if the script is executed directly