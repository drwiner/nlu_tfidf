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
normalized types are surrounded by \< \>

Stopword function from sklearn tfidf directly. 


#### david.winer@kasisto.com
