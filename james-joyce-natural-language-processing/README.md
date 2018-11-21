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

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

### Download Data

1. Project Gutenberg does not allow automated downloads. Manually download the following literary works in UTF-8 format from Project Gutenberg and save the text files in the `/data/raw/` directory. The download links below lead directly to the UTF-8 text version of the literary works. 
  1. [*Dubliners*](http://www.gutenberg.org/files/2814/2814-0.txt)
  2. [*A Portrait of the Artist as a Young Man*](http://www.gutenberg.org/files/4217/4217-0.txt)
2. Manually download the NRC Emotion Lexicon. The license for the lexicon does not allow for the redistribution of the lexicon. The download link below leads directly to the zipped lexicon.
  1. [NRC Emotion Lexicon](http://sentiment.nrc.ca/lexicons-for-research/NRC-Emotion-Lexicon.zip)

### Requirements

1. Python 3.6
2. Graphviz 2.40.1 (http://graphviz.org)
  1. The version of Graphviz used for this project was installed on MacOS 10.13.6 via Homebrew 1.7.7.
  1. The PyGraphviz package, which is listed in `requirements.txt`, depends on Graphviz. When installing PyGraphviz, you might need to specify the location of the Graphviz library on your local system, depending on which method you used to install Graphviz.
  2. For example: `pip install pygraphviz==1.5 --install-option="--include-path=/usr/local/Cellar/graphviz" --install-option="--library-path=/usr/local/Cellar/graphviz"`
3. Install project requirements using `pip install -r requirements.txt`.

### Notebooks Execution Order

1. clean-and-preprocess-data.ipynb
2. natural-language-processing.ipynb
