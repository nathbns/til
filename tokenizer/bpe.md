## BPE(byte-pair encoding): Use in GPT | GPT2

```python
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained('gpt2')

from collections import defaultdict

corpus = ["This is an introduction course.", "This is about tokenization.", "This section shows multiple tokenizer algorithms.", "Hope you like the content so far."]

# calculate the freq foreach words in the corpus

word_freqs = defaultdict(int)

for txt in corpus:
    word_with_offset = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt)
    new_word = [word for word, _ in word_with_offset]
    for w in new_word:
        word_freqs[w] += 1

print(word_freqs)

# vocab base on the corpus (set())

# word_freqs.keys()
alphabet = []

for word in word_freqs:
    for c in word:
        if c not in alphabet:
            alphabet.append(c)
        
alphabet.sort()
print(f"alphabet: {alphabet}\n")

# we add the 'special' char -> "<|endoftext|>"
vocab = ["<|endoftext|>"] + alphabet.copy()
print(f"vocab: {vocab}")

# for training we need foreach word to decompose in each char
splits = {word: [c for c in word] for word in word_freqs.keys()}

# function calcule freq of each pairs

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

# just checking our new dict
# compute_pairs_freq(split)
pair_freqs = compute_pair_freqs(splits)

for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

# finding the best pair
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        max_freq = freq
        best_pair = pair

print(f"best_pair = {best_pair} && max_freq = {max_freq}")

# So the first fusion to learn is ==> ('Ġ', 't') -> 'Ġt' && we add 'Ġt' to the vocab

# we have to merge the pair learn by our BPE tokenizer

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a + b] + split[i+2:]
            else:
                i += 1
        splits[word] = split
    return splits


target_vocab_size = 100
merges = {}

while len(vocab) < target_vocab_size:
    # calculate the pair freq
    pair_freqs = compute_pair_freqs(splits)
    # find the best freq pair
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            max_freq = freq
            best_pair = pair
    # merging the pair in the corpus
    splits = merge_pair(*best_pair, splits)
    
    merges[best_pair] = best_pair[0] + best_pair[1]
    
    # add the new vocab
    vocab.append(best_pair[0] + best_pair[1])

print(merges)
print(vocab)

"""
{('i', 's'): 'is', ('t', 'i'): 'ti', ('o', 'n'): 'on', ('T', 'h'): 'Th', ('Th', 'is'): 'This', ('Ġ', 'a'): 'Ġa', ('ti', 'on'): 'tion', ('o', 'u'): 'ou', ('k', 'e'): 'ke', ('Ġ', 's'): 'Ġs', ('Ġ', 'is'): 'Ġis', ('n', 't'): 'nt', ('c', 'tion'): 'ction', ('Ġ', 'c'): 'Ġc', ('Ġt', 'o'): 'Ġto', ('Ġto', 'ke'): 'Ġtoke', ('Ġtoke', 'n'): 'Ġtoken', ('Ġtoken', 'i'): 'Ġtokeni', ('Ġtokeni', 'z'): 'Ġtokeniz', ('Ġa', 'n'): 'Ġan', ('Ġ', 'i'): 'Ġi', ('Ġi', 'nt'): 'Ġint', ('Ġint', 'r'): 'Ġintr', ('Ġintr', 'o'): 'Ġintro', ('Ġintro', 'd'): 'Ġintrod', ('Ġintrod', 'u'): 'Ġintrodu', ('Ġintrodu', 'ction'): 'Ġintroduction', ('Ġc', 'ou'): 'Ġcou', ('Ġcou', 'r'): 'Ġcour', ('Ġcour', 's'): 'Ġcours', ('Ġcours', 'e'): 'Ġcourse', ('Ġa', 'b'): 'Ġab', ('Ġab', 'ou'): 'Ġabou', ('Ġabou', 't'): 'Ġabout', ('Ġtokeniz', 'a'): 'Ġtokeniza', ('Ġtokeniza', 'tion'): 'Ġtokenization', ('Ġs', 'e'): 'Ġse', ('Ġse', 'ction'): 'Ġsection', ('Ġs', 'h'): 'Ġsh', ('Ġsh', 'o'): 'Ġsho', ('Ġsho', 'w'): 'Ġshow', ('Ġshow', 's'): 'Ġshows', ('Ġ', 'm'): 'Ġm', ('Ġm', 'u'): 'Ġmu', ('Ġmu', 'l'): 'Ġmul', ('Ġmul', 'ti'): 'Ġmulti', ('Ġmulti', 'p'): 'Ġmultip', ('Ġmultip', 'l'): 'Ġmultipl', ('Ġmultipl', 'e'): 'Ġmultiple', ('Ġtokeniz', 'e'): 'Ġtokenize', ('Ġtokenize', 'r'): 'Ġtokenizer', ('Ġa', 'l'): 'Ġal', ('Ġal', 'g'): 'Ġalg', ('Ġalg', 'o'): 'Ġalgo', ('Ġalgo', 'r'): 'Ġalgor', ('Ġalgor', 'i'): 'Ġalgori', ('Ġalgori', 't'): 'Ġalgorit', ('Ġalgorit', 'h'): 'Ġalgorith', ('Ġalgorith', 'm'): 'Ġalgorithm', ('Ġalgorithm', 's'): 'Ġalgorithms', ('H', 'o'): 'Ho', ('Ho', 'p'): 'Hop', ('Hop', 'e'): 'Hope', ('Ġ', 'y'): 'Ġy', ('Ġy', 'ou'): 'Ġyou', ('Ġ', 'l'): 'Ġl', ('Ġl', 'i'): 'Ġli', ('Ġli', 'ke'): 'Ġlike', ('Ġt', 'h'): 'Ġth', ('Ġth', 'e'): 'Ġthe', ('Ġc', 'on'): 'Ġcon', ('Ġcon', 't'): 'Ġcont', ('Ġcont', 'e'): 'Ġconte'}
['<|endoftext|>', '.', 'H', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'y', 'z', 'Ġ', 'is', 'ti', 'on', 'Th', 'This', 'Ġa', 'tion', 'ou', 'ke', 'Ġs', 'Ġis', 'nt', 'ction', 'Ġc', 'Ġto', 'Ġtoke', 'Ġtoken', 'Ġtokeni', 'Ġtokeniz', 'Ġan', 'Ġi', 'Ġint', 'Ġintr', 'Ġintro', 'Ġintrod', 'Ġintrodu', 'Ġintroduction', 'Ġcou', 'Ġcour', 'Ġcours', 'Ġcourse', 'Ġab', 'Ġabou', 'Ġabout', 'Ġtokeniza', 'Ġtokenization', 'Ġse', 'Ġsection', 'Ġsh', 'Ġsho', 'Ġshow', 'Ġshows', 'Ġm', 'Ġmu', 'Ġmul', 'Ġmulti', 'Ġmultip', 'Ġmultipl', 'Ġmultiple', 'Ġtokenize', 'Ġtokenizer', 'Ġal', 'Ġalg', 'Ġalgo', 'Ġalgor', 'Ġalgori', 'Ġalgorit', 'Ġalgorith', 'Ġalgorithm', 'Ġalgorithms', 'Ho', 'Hop', 'Hope', 'Ġy', 'Ġyou', 'Ġl', 'Ġli', 'Ġlike', 'Ġth', 'Ġthe', 'Ġcon', 'Ġcont', 'Ġconte']
""" 
```