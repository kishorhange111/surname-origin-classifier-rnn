import torch
import unicodedata
import string

alldata = ['Dutch', 'Arabic', 'German', 'Japanese', 'Irish', 'Polish', 'Korean', 'French', 'Greek', 'Portuguese', 'Vietnamese', 'Russian', 'Czech', 'Chinese', 'Spanish', 'Scottish', 'English', 'Italian']

allowed_characters = string.ascii_lowercase + " .,;'" + "_"
n_letters = len(allowed_characters)

def unicode_to_ascii(s):
    return ''.join(
        c.lower() for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )


def letter_to_index(letter):
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    else:
        return allowed_characters.find(letter)


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def label_from_output(predicted_output, output_labels):
    top_n, top_i = predicted_output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i