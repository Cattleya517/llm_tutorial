import torch

def get_words():
    words = open('names.txt', 'r').read().splitlines()
    return words

for w in get_words()[:3]:
    for ch1, ch2 in zip (w, w[1:]):
        print(ch1, ch2)
        list(w)