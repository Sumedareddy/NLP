### Sentiment Analysis with RNN and Word Embeddings

This project involves training a Recurrent Neural Network (RNN) for sentiment analysis using word embeddings generated from different techniques, namely Word2Vec and FastText. The dataset used for training and evaluation is sourced from the Stanford AI Lab.

#### Dataset
- **Dataset Link:** [Stanford Sentiment Analysis Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Description:** This dataset contains movie reviews labeled with sentiment (positive or negative). We will use the "train" folder for training word embeddings and sentiment analysis, and the "test" folder for evaluating the sentiment classification task.

#### Implementation Steps
1. **Word Embeddings Generation:**
   - Implement the Word2Vec model using the skip-gram architecture with negative sampling.
   - Implement the FastText model to train word vectors.

2. **Sentiment Classification with RNN:**
   - Utilize the trained word vectors from Word2Vec and FastText to perform sentiment classification using RNN.
   - Train and evaluate the RNN model on the sentiment analysis task.

3. **Results and Analysis:**
   - Present experimental results including performance metrics (e.g., accuracy, F1 score) for each word embedding technique.
   - Visualize and compare results using tables and graphs.
   - Discuss findings and insights from the experiments.

#### Repository Contents
- **`data/`**: Contains the downloaded dataset. The "train" folder will be used for training word embeddings and sentiment analysis.
- **`src/`**: Source code for implementing Word2Vec, FastText, and RNN sentiment analysis.
  - `word2vec.py`: Implementation of Word2Vec model.
  - `fasttext.py`: Implementation of FastText model.
  - `rnn_sentiment.py`: RNN model for sentiment analysis using word embeddings.
- **`README.md`**: Instructions, project overview, and documentation.
- **`requirements.txt`**: List of Python dependencies needed to run the project.

#### How to Run
1. **Setup Environment:**
   - Install required packages using `pip install -r requirements.txt`.
   - Download the dataset and place it in the `data/` directory.

2. **Generate Word Embeddings:**
   - Run `word2vec.py` to train Word2Vec embeddings.
   - Run `fasttext.py` to train FastText embeddings.

3. **Sentiment Classification:**
   - Execute `rnn_sentiment.py` to train and evaluate RNN sentiment analysis using the pre-trained word embeddings.

#### Analysis and Results
- The `results/` directory will contain output files, logs, and visualizations.
- Analyze and compare the performance of RNN using Word2Vec and FastText embeddings.
- Discuss experimental findings, including hyperparameters used and their impact on results.

#### References
1. https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html#transforming-
rnn-outputs
2. https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/.

For more details on the implementation and experimental setup, please refer to the project source code and accompanying documentation.

--- 

