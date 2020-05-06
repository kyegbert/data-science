# Detecting Literary Themes in James Joyce's Works Using Natural Language Processing

## Project Description
Can contemporary Natural Language Processing (NLP) techniques accurately detect early twentieth century literary themes? Let's find out by using the NLP techniques of sentiment analysis, TF-IDF weighting, topic modeling and bigram networks on the works of the early twentieth century Irish writer James Joyce. Two of Joyce's early major works, the short story collection *Dubliners* (1914) and the novel *A Portrait of the Artist as a Young Man* (1916), are set in Dublin, Ireland and focus on themes of identity, religion, death, exile and Irish nationalism. 

How well do these NLP techniques surface words associated with this collection of topics?

### Methods Used
* Natural Language Processing
* Sentiment Analysis
* Term Frequency – Inverse Document Frequency
* Topic Modeling
* Latent Dirichlet Allocation
* Non-negative Matrix Factorization
* Bigram Networks
* Markov Chain
* Data Preprocessing
* Data Visualization

### Technologies
* Python
* Jupyter Notebooks
* Docker

## Getting Started

1. Follow the Getting Started instructions in the root `README.md`.

### Download Data

1. Project Gutenberg does not allow automated downloads. Manually download the following literary works in UTF-8 format from Project Gutenberg and save the text files in the `/data/raw/` directory. The download links below lead directly to the UTF-8 text version of the literary works. 
  1. [*Dubliners*](http://www.gutenberg.org/files/2814/2814-0.txt)
  2. [*A Portrait of the Artist as a Young Man*](http://www.gutenberg.org/files/4217/4217-0.txt)
2. Manually download the NRC Emotion Lexicon. Copy the `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` file to the `raw` directory. The license for the lexicon does not allow for the redistribution of the lexicon. The download link below leads directly to the zipped lexicon.
  1. [NRC Emotion Lexicon](http://sentiment.nrc.ca/lexicons-for-research/NRC-Emotion-Lexicon.zip)

### Notebooks Execution Order

Run the notebooks in the following order:

1. 1-clean-and-preprocess-data.ipynb
2. 2-natural-language-processing.ipynb
