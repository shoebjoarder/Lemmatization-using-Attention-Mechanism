import torch
import spacy
import csv
from tqdm import tqdm
import collections
from textblob import TextBlob
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
import sys
import pickle
import stanza
import re

stanza.download('en')
nlp = stanza.Pipeline('en')
spacyTokenizer = spacy.load("en")


def process_sentence(model, sentence, inputText, outputText, device, max_length=50):
    # Load tokenizer
    spacy_tokenizer = spacy.load("en")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, inputText.init_token)
    tokens.append(inputText.eos_token)

    # Go through each token and convert to an index
    text_to_indices = [inputText.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [outputText.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == outputText.vocab.stoi["<eos>"]:
            break

    processed_sentence = [outputText.vocab.itos[idx] for idx in outputs]
    # remove start token
    return processed_sentence[1:]


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open("./model/" + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def evaluate(data, predict):
    positive = 0
    negative = 0
    for i in range(len(data)):
        x = data[i]
        y = predict[i]
        x.sort()
        y.sort()
        if x == y:
            positive = positive + 1
        else:
            negative = negative + 1
    score = positive / (positive + negative)
    return score


def tokenizer(text):
    return [token.text for token in spacyTokenizer.tokenizer(text)]


def predict(dataInput, dataTarget, model, inputText, outputText, device):
    outputs = []
    targets = []
    for i in tqdm(range(len(dataInput))):
        prediction = process_sentence(model, dataInput[i], inputText, outputText, device)
        prediction = prediction[:-1]  # remove <eos> token
        # prediction = " ".join(prediction)
        outputs.append(prediction)
        targets.append(tokenizer(dataTarget[i]))
    return outputs, targets


def parsingCorpus(df, fileName1, fileName2):
    sentence = ""
    lemma = ""
    for iter in tqdm(range(len(df))):
        try:
            word = str(df['Section'][iter])
            lem = str(df['section'][iter])
            if word == ".":
                sentence = sentence + word + "\n"
                lemma = lemma + lem + "\n"
            elif word == "<p>":
                sentence = sentence + "\n"
                lemma = lemma + "\n"
            else:
                sentence = sentence + str(word) + " "
                lemma = lemma + str(lem) + " "
        except KeyError:
            continue

    print("\n\nDone sentences & lemma!")

    file1 = "./dataset/" + fileName1 + ".txt"
    file2 = "./dataset/" + fileName2 + ".txt"
    writeFile(file1, sentence)
    writeFile(file2, lemma)


def writeFile(filename, var):
    with open(filename,'w') as file:
        for line in var:
            file.write(line)


def readFile(file):
    filename = "./dataset/" + file
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def text_cleaner(text):
    # Remove punctuation marks
    text_blob = TextBlob(text)
    text = ' '.join(text_blob.words)

    # Remove emojis
    eng_char = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = ''.join(c for c in text if c in eng_char)

    # Remove extra spaces
    text = re.sub(' +', ' ', text)

    return text