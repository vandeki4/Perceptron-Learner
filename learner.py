import numpy as np
# One-hot documents are sparse, so use sparse row vectors to make operations faster
from scipy.sparse import csr_matrix
from tqdm import tqdm # progress bar

from sklearn.datasets import fetch_20newsgroups

def onehot_document(document, corpus, token_lookup):
    '''Convert a document into a onehot vector using a preexisting dictionary'''

    data = []
    indices = []
    col_ind = []

    for word in set(document.split()):
        if word in token_lookup:
            idx = token_lookup[word]
            data.append(1.0)
            indices.append(idx)
    
    return csr_matrix((data, indices, [0, len(indices)]), shape=(1, len(corpus)))

if __name__ == '__main__':
    # Load dataset
    train_dataset = fetch_20newsgroups(subset='train')
    test_dataset = fetch_20newsgroups(subset='test')

    # Class_number (0-19 integer) -> Class name (string)
    class_name = train_dataset.target_names

    # List of tokens seen in the trainset
    corpus = []
    # Token -> index in the corpus, backwards lookup
    token_lookup = {}

    # Preprocess into one-hot vectors of word presence separated by whitespace
    # No removing stopwords, punctuation, etc, naive but straightforward

    # First tokenize the trainset
    for document in train_dataset.data:
        for token in document.split():
            if token not in token_lookup:
                token_lookup[token] = len(corpus)
                corpus.append(token)


    # I'm using a classifier for each category, so there will be 20 perceptrons.
    weights_for_each_perceptron = np.zeros((20, len(corpus)))
    activation = lambda x: 1.0 / (1.0 + np.exp(-x)) # sigmoid
    learning_rate = 1e-2

    # Iterate over each document in the train set, using a progress bar
    for document, class_num in tqdm(
            zip(train_dataset.data, train_dataset.target), 
            total=len(train_dataset.data)):
        onehot_encoding = onehot_document(document, corpus, token_lookup)

        # For each of the twenty perceptrons:
        for i, weights in enumerate(weights_for_each_perceptron):
            label = 1.0 if (class_num == i) else 0.0

            pred = activation(onehot_encoding.dot(weights))
            
            # Step in the direction of negative gradient
            # This loop exploits the sparse structure of the one_hot array to accelerate the operation.
            for j, val in zip(onehot_encoding.indices, onehot_encoding.data):
                grad = (pred - label) * val
                weights_for_each_perceptron[i, j] += -grad * learning_rate

    # Calculate the confusion matrix for this classifier
    confusion = np.zeros((20, 20), dtype=np.int32)
    for test_document, class_num in tqdm(
            zip(test_dataset.data, train_dataset.target),
            total=len(test_dataset.data)):
        onehot_encoding = onehot_document(test_document, corpus, token_lookup)

        # Predict the label by choosing the maximum logit...
        pred = np.argmax(onehot_encoding.dot(weights_for_each_perceptron.T))
        confusion[class_num, pred] += 1
    
    np.savetxt('confusion.txt', confusion / np.sum(confusion, axis=1, keepdims=True), header=' '.join(class_name))

    # Write the per-class accuracy
    with open('per-class-acc.txt', 'w') as f:
        for class_num, pred_counts in enumerate(confusion):
            acc = pred_counts[class_num] / np.sum(pred_counts)
            f.write(f'{class_name[class_num]}: {acc}\n')