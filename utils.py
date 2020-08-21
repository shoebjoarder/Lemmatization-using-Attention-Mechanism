import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys
import pickle


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