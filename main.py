# main.py
# Entry point for the text error correction project.
# This script loads data, cleans it, detects errors, and corrects them using the provided models.

from config import ConfigManager
from dataset_parser import DatasetParser
from data_utils import clean_text_all
from error_corrector_runner import ErrorCorrectionRunner
from error_detection_model_runner import ErrorDetectionModelRunner
from generate_synthetic_data import generate_synthetic_data

def main():
    # Load configuration settings
    config = ConfigManager()

    # Initialize DatasetParser with the configuration
    parser = DatasetParser(config)

    # Read and preprocess training data
    train_words, _, train_gs_words, train_labels = parser.read_train_dataset()
    input_texts = []
    target_texts = []

    for sentence, gs_sentence, label in zip(train_words, train_gs_words, train_labels):
        for w, gs, l in zip(sentence, gs_sentence, label):
            input = clean_text_all(w)
            target = clean_text_all(gs)
            if len(input) > 25 or len(target) > 25:
                continue
            target = '\t' + target + '\n'
            input_texts.append(input)
            target_texts.append(target)

    # Generate synthetic data if needed
    generate_synthetic_data(config)

    # Run error detection models
    detection_runner = ErrorDetectionModelRunner()
    detection_runner.run()

    # Run error correction models
    correction_runner = ErrorCorrectionRunner()
    correction_runner.run()

if __name__ == "__main__":
    main()