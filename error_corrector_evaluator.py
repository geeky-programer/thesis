import torch
import nltk.data

from config import ConfigManager
from tqdm import tqdm
from error_detection_model_runner import ErrorDetectorRunner
from error_corrector_runner import ErrorCorrectionRunner
from dataset_parser import DatasetParser
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

class ErrorCorrectionEvaluator:
    def __init__(self, config):
        self.config = config

        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Available device: {}".format(self.device))

        # Initialize error detector runner
        self.edrunner = ErrorDetectorRunner(self.config)
        self.tokenizer = self.edrunner.tokenizer
        self.model = self.edrunner.model

        # Initialize dataset parser
        self.dataset_parser = DatasetParser(config)

        # Download NLTK punkt tokenizer
        nltk.download('punkt')
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Initialize error corrector runner
        self.error_corrector = ErrorCorrectionRunner(config=config,
                                                     enable_diagonal_attention_loss=False,
                                                     enable_coverage=False,
                                                     enable_copy=False)

        self.correction_util = self.error_corrector.correction_util

        # Set batch size for data loader
        self.batch_size = 64

        # Load error corrector model and features
        self.error_corrector.load()
        self.features = self.error_corrector.features

    def evaluate(self):
        # Read test dataset
        test_words, _, test_gs_words, _ = self.dataset_parser.read_test_dataset()
        wrong = []
        gs = []

        # Iterate over test words and ground truth words
        for x, y in tqdm(zip(test_words, test_gs_words)):
            _, l = self.edrunner.inference(x)

            # Skip if lengths do not match
            if len(l) != len(y):
                continue

            # Collect wrong words and corresponding ground truth
            for tok, g, lb in zip(x, y, l):
                if lb == 1 and tok != '[UNK]':
                    wrong.append(tok)
                    gs.append(g)

        i, t = [], []
        # Prepare input and target sequences for correction
        for w, gs in zip(wrong, gs):
            if len(w) >= self.features.max_encoder_sequence_length or len(gs) >= self.features.max_decoder_sequence_length:
                continue

            target = self.correction_util.start_character + gs + self.correction_util.end_character

            i.append(w)
            t.append(target)

        # Create data loader for test data
        loader = self.create_test_dataloader(i, t)
        self.error_corrector.corrector.eval_loader(loader)

    def create_test_dataloader(self, input_test, target_test):
        # Transform input and target sequences to IDs
        inputs_test_transformed = self.error_corrector.encoder_text_processor.to_ids(input_test)
        targets_test_transformed = self.error_corrector.decoder_text_processor.to_ids(target_test)

        # Convert numpy arrays to torch tensors
        input_test_torch = torch.from_numpy(inputs_test_transformed)
        target_test_torch = torch.from_numpy(targets_test_transformed)

        # Create TensorDataset and DataLoader
        test_loader = TensorDataset(input_test_torch, target_test_torch)
        test_sampler = RandomSampler(test_loader)
        test_dataloader = DataLoader(test_loader, pin_memory=True, num_workers=2, sampler=test_sampler, batch_size=self.batch_size)

        return test_dataloader

def main():
    # Initialize configuration manager
    config = ConfigManager()
    
    # Create evaluator instance and run evaluation
    evaluator = ErrorCorrectionEvaluator(config)
    evaluator.evaluate()

if __name__ == '__main__':
    main()