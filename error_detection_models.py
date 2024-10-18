from data_utils import PUNCTUATION_ALL
from torch import nn
from transformers import BertModel, DistilBertModel, XLMRobertaModel, XLMModel, RobertaModel, DebertaV2Model
import fasttext

class ErrorDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = 2  # Number of labels for classification (binary classification)

        # Uncomment the respective pretrained model you want to use
        # self.pretrained_model = BertModel.from_pretrained("distilbert-base-multilingual-cased")
        # self.pretrained_model = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base")
        # self.pretrained_model = RobertaModel.from_pretrained("iarfmoose/roberta-base-bulgarian")
        # self.pretrained_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        # self.pretrained_model = XLMModel.from_pretrained("xlm-mlm-100-1280")
        self.pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")  # Default pretrained model
        self.embedding_dim = 768  # Embedding dimension size
        
        # Linear classifier layer
        self.classifier = nn.Linear(self.embedding_dim, self.num_labels)

    def forward(self, input, attention_mask):
        # Get the output from the pretrained model
        outputs = self.pretrained_model(input, attention_mask)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, embedding_dim)

        # Pass the output through the classifier
        return self.classifier(sequence_output)
    

class CladaErrorDetectionModel:
    def __init__(self, config) -> None:
        self.config = config
        # Load the CLADA dictionary and remove punctuation and newlines
        self.clada_dictionary = set(line.strip(PUNCTUATION_ALL + '\n').lower() for line in open(self.config.get_clada_dictionary(), encoding='utf-16', mode="r"))
    
    def forward(self, input):
        # Check if the input is in the CLADA dictionary
        return input not in self.clada_dictionary  # Labels: 0 - correct, 1 - incorrect
    
class FastTextErrorDetectionModel:
    def __init__(self, config) -> None:
        self.config = config
        self.model = None  # Initialize the FastText model as None

    def forward(self, input):
        if self.model is None:
            raise Exception("Fast Text Model is none. Need to train it first.")
        
        # Predict the label using the FastText model
        lbl, p = self.model.predict(input)
        predicted_lbl = int(lbl[0].replace("__label__", ""))  # Extract the label

        return predicted_lbl
    
    def train(self, fast_text_file, epoch=1000):
        # Train the FastText model with the provided file and number of epochs
        self.model = fasttext.train_supervised(fast_text_file, epoch=epoch)