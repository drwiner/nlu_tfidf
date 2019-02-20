# TF-IDF
Term Frequency - Inverse Document Frequency

Uses sklearn.

## Which features are most distinctive to each label?

Higher tf-idf score, higher distinctiveness

## Source
Training features, used by Mallet data importer. They are created right before training, and end with .norm

### N=1
Single tokens evaluated on basis of tf-idf. For Mallet, this could be n-gram of size 2, e.g. how_do

### N=2
All pairs of tokens that appear in a single training utterance. If how_do and can_i are tokens, then when N=2, can_i_how_do appears as a token as input to tf-idf. Single tokens only appear if they appear in isolation for a single training utterance. 

### N=3

All triplets of tokens that appear in a single training utterance, where doublets are only allowed if they appear in isolation.

### Notation
Replacements are made in the text. For each pair below, the first item is replaced by the second item:


[("<", "f_"), (">", "_f"), ("'", "_apo_"), ("-", "_hyp_"), (".", "_dot_"), (",", "_ca_"), ("\"", "_quo_"), ("“", "_quo_"), ("\'", "_apo_"), (")", '_rp_'), ("(", "_lp"), ("/", "_fsl_"), ("\\", "_bsl_"), ("=", "_eq_"), ("@", "_aat_"), ("$", "_dol_"), ("%", "_per_"), ("*", "_star_"), ("#", "_hash_"), ("+", "_plus_"), ("^", "_car_"), ("!", "_excl_"), ("?", "_quest_"), ("&", "_aand_")]


Thus, features are surrounded by \_f\_feat\_f\_ rather than \<feat\>. This is done to still use sklearn's tfidf without modifying much of code. I also removed stopword function from sklearn tfidf directly. 

### Results
all --> first-level intent classifier results
getAnswer --> tfidf results for getAnswer
getDefinition --> tfidf results for getDefinition

#### david.winer@kasisto.com
