The goal of this repository is to collect **reproducable** benchmarks for NLP tasks using standard data sets. Unless otherwise noted, all models are implemented in Tensorflow.

### Why?

Reproducability is a major problem in NLP research. Many times have I talked to researchers who could not reproduce the numbers reported by other researchers. Even though the Deep Learning community is more open and transparent than most other communities, many don't publish their code. Even if code is published, it is often difficult to compare techniques directly, because the implemntation framework or data pre-processing pipeline is not the same. The goal of this repository is to implement popular Deep Learning models and evaluate them on standard data sets. The benchmarks and models here can serve as a entry point for new researchers to implement and compare their own models. 


### Data Sets

- **Movie Reviews (MR)**: Movie Reviews from Rotten Tomaties. 5331 positive and 5331 negative processed sentences / snippets. ([Source](http://www.cs.cornell.edu/people/pabo/movie-review-data/))
- **Stanford Sentiment Treebank (SST)**:  variation on the MR dataset with individual subphrases tagged on Mechanical Turk. The Stanford Parser is used to parses all 10,662 sentences. In approximately 1,100 cases it splits the snippet into multiple sentences. We then used Amazon Mechanical Turk to label the resulting 215,154 phrases. This dataset was first used in [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) ([Source](http://nlp.stanford.edu/sentiment/))
- **IMDB Movie Reviews (IMDB)**: TODO ([Source](http://ai.stanford.edu/~amaas/data/sentiment/))
- **20 Newsgroups**: TODO ([Source](http://qwone.com/~jason/20Newsgroups/))