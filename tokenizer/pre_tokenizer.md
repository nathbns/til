# Pre-Tokenizer:
- A tokenizer cannot be trained on raw text alone. 
- Instead, we first need to split the texts into small entities, like words. 

- A word-based tokenizer can simply split a raw text into words on whitespace and punctuation. Those words will be the boundaries of the subtokens the tokenizer can learn during its training.

```python
your_name = "Nathan" # replace with your own name :)
sentence = f"Hello my name is {your_name}, i'm a french computer science student."

from transformers import AutoTokenizer
# to simplify some the call of our model, we will use these functions.

def tokenizer_function(model_checkpoint):
    return AutoTokenizer.from_pretrained(model_checkpoint)

#
def tokenizer_pre_tokenize_str(tokenizer, sentence):
    return tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)

model_checkpoint = 'bert-base-uncased'
tokenizer = tokenizer_function(model_checkpoint)

pre_tokenize_sentence = tokenizer_pre_tokenize_str(tokenizer, sentence)
```

We can see that with gpt2, each 'new' word start with 'Ġ'.

### Let's try another model again.

```python
model_checkpoint = 'gpt2'
tokenizer = tokenizer_function(model_checkpoint)

pre_tokenize_sentence = tokenizer_pre_tokenize_str(tokenizer, sentence)
s = ""
for i, (word, offset) in enumerate(pre_tokenize_sentence):
    s += word
    if i < len(pre_tokenize_sentence) - 1:
        s += ', '
print(s) # -> "Hello, Ġmy, Ġname, Ġis, ĠNathan, ,, Ġi, 'm, Ġa, Ġfrench, Ġcomputer, Ġscience, Ġstudent, ."
```

### In conclusion, each model is train to pre-tokenize differently.
#### Now that we see Normalization and Pre-Tokenization, we can reach the next step of the tokenizer.