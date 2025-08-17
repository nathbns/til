# Normalization : 
- cleanup the text (rm accents | spaces | unicode normalization | and others...) 


```python
your_name = "Nathan" # replace with your own name :)
sentence = f"Hellô my nAme is {your_name}, i'm á freñch computer scIence student."

from transformers import AutoTokenizer

# we will use this model, 
# because we can comparate easily with his alt model 'bert-base-cased'
model_checkpoint = 'bert-base-uncased' 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# the next line of code, show applied the normalization on our 'sentence'
print(f"{tokenizer.backend_tokenizer.normalizer.normalize_str(sentence)}\n")

model_checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print(f"{tokenizer.backend_tokenizer.normalizer.normalize_str(sentence)}\n")
```

We see that this model does not applied lowercasing or removing accents.


### Each model normalize with his own rule. If you use one, it's you responsability to check the documentation if available.