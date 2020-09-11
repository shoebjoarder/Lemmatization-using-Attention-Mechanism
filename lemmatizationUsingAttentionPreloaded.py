#!/usr/bin/env python
# coding: utf-8

# # Lemmatization using Attention Mechanism Preloaded Model

# In[1]:


# Import library
import random
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import spacy
import pickle
import csv
from utils import process_sentence, load_obj, save_obj, save_checkpoint, load_checkpoint, predict, evaluate
from torchtext.data import Field, BucketIterator, TabularDataset, Dataset
from sklearn.model_selection import train_test_split


# In[2]:


spacyTokenizer = spacy.load("en")

def tokenize_input(text):
    return [token.text for token in spacyTokenizer.tokenizer(text)]

def tokenize_output(text):
    return [token.text for token in spacyTokenizer.tokenizer(text)]


# In[3]:


inputText = Field(init_token="<sos>", eos_token="<eos>", tokenize=tokenize_input, lower=True)
outputText = Field(init_token="<sos>", eos_token="<eos>", tokenize=tokenize_output, lower=True)


# In[4]:


fields = {"InputText": ("inputText", inputText), "OutputText": ("outputText", outputText)}


# In[5]:

# Load the pretrained vocabulary
inputText.vocab = load_obj("inputText")
outputText.vocab = load_obj("outputText")


# In[6]:


class Transformer(nn.Module):
    '''This is the transformer model that has been used to create the lemmatization using attention mechanism'''
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, inputText):
        src_mask = inputText.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, inputText, outputText):
        src_seq_length, N = inputText.shape
        trg_seq_length, N = outputText.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(inputText) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(outputText) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(inputText)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


# In[7]:


# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = True

# Training hyperparameters
num_epochs = 10
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(inputText.vocab)
trg_vocab_size = len(outputText.vocab)
embedding_size = 512    # default: 512
num_heads = 8
num_encoder_layers = 3  # 6 in paper
num_decoder_layers = 3
dropout = 0.10
max_len = 100           # max_len=70 for old model
forward_expansion = 4
src_pad_idx = inputText.vocab.stoi["<pad>"]

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0


# In[8]:


model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)


# In[9]:


optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[10]:


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)


# In[11]:


pad_idx = inputText.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)


# In[12]:

# Loading the pretrained model
if load_model:
    load_checkpoint(torch.load("./training/data/model/my_checkpoint.pth.tar"), model, optimizer)


# In[13]:

# Example sentence
src = "The feasibility study estimates that it would take passengers about four minutes to cross the Potomac River on the gondola."

prediction = process_sentence(model, src, inputText, outputText, device)
prediction = prediction[:-1]  # remove <eos> token

print(prediction)


# ## Evaluation

# In[14]:


# Loading the test dataset
test_data = pd.read_csv("./training/test/test-sample.txt", header=0, names=['InputText', 'OutputText'], sep='\t', encoding='utf-8')


# In[15]:


# Making predictions of lemma using the transformer model
predictions, targets = predict(test_data["InputText"], test_data["OutputText"], model, inputText, outputText, device)


# In[16]:


# Evaluation of the lemmatization using mechanism
score = evaluate(targets, predictions)
print(f"Attention Accuracy {score * 100:.2f}")


# ## Evaluation with Stanza

# In[17]:


import stanza
stanza.download('en')

nlp = stanza.Pipeline(processors = "tokenize,mwt,lemma,pos")


# In[18]:

# Function to return a list of all the stanza's lemmatized sentences
def extract_lemma(df):
    prediction = []
    for iter in tqdm(range(len(df))):
        doc = nlp(df[iter])
        for sent in doc.sentences:
            lemma = []
            for wrd in sent.words:
                lemma.append(str(wrd.lemma).lower())
                # word.append(str(wrd.text))
            prediction.append(lemma)
            # target.append(word)
        #return a dataframe
    return prediction


# In[19]:

# Function to predict the lemma of a sentence
def predictionStanza(data):
    prediction = []
    for iter in tqdm(range(len(data))):
        doc = nlp(data[iter])
        text = ""
        for sent in doc.sentences:
            for wrd in sent.words:
                text = text + wrd.lemma + " "
        lemma = text.split()
        prediction.append(lemma)
    return prediction


# In[20]:


predictStanza = predictionStanza(test_data["InputText"])


# In[21]:

# Evaluation of the stanza model
score = evaluate(targets, predictStanza)
print(f"Stanza Accuracy {score * 100:.2f}")
