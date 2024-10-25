from transformers import AutoTokenizer
from gensim import corpora
import gensim
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def lda_preprocess(data_tokenized, id2word=None):
    if id2word is None:
        id2word = corpora.Dictionary(data_tokenized)

    processed_texts = data_tokenized
    corpus = [id2word.doc2bow(text) for text in processed_texts]

    return corpus, id2word, processed_texts

# Set hyperparameters
num_epochs = 10

# Define BERT tokenizer
model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the texts
train_sentences = pd.read_csv("data/train.csv", encoding='utf-8')["text"].tolist()
test_sentences = pd.read_csv("data/test.csv", encoding='utf-8')["text"].tolist()

combined_sentences = train_sentences + test_sentences

# Perform LDA topic modeling
tokenized = [tokenizer.tokenize(sentence) for sentence in combined_sentences]
corpus, id2word, _ = lda_preprocess(tokenized)

# Create a file to save the results
output_file = open("lda_results.txt", "a")

# Iterate over different number of topics
for num_topics in range(10, 201, 10):
    # Train LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, passes=10)

    # Compute perplexity
    perplexity = lda_model.log_perplexity(corpus)

    # Compute coherence
    coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=tokenized, dictionary=id2word)
    coherence = coherence_model_lda.get_coherence()

    print(f"Number of Topics: {num_topics}")
    print(f"Perplexity: {perplexity}")
    print(f"Coherence: {coherence}")
    print("\n")

    # Save the results to the output file
    output_file.write(f"Number of Topics: {num_topics}\n")
    output_file.write(f"Perplexity: {perplexity}\n")
    output_file.write(f"Coherence: {coherence}\n")
    output_file.write("\n")

# Close the output file
output_file.close()
