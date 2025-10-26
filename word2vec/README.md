**Word2Vec Skip-Gram with Toy Corpus**

## Implementation Steps
- Built a Word2Vec skip-gram model from scratch using a small, custom corpus.
- Explored different model configurations: embedding size, negative sampling, and multi-layer architectures.
- Compared multiple versions of word embeddings (v1, v2, v3) using PCA visualization.
- Used a diverse set of words (people, places, scientific terms) for embedding comparison.


## Experiments & Findings
- Increasing embedding size and training epochs improved word relationships.
- Negative sampling helped the model learn better discriminative features.
- Multi-layer architectures captured more complex word associations.
- PCA plots revealed how different model settings affect word clustering and similarity.
- Found that a larger, more diverse corpus leads to higher-quality embeddings.
- Noted that some words were missing in certain embeddings due to corpus limitations.
- Compared word pairs (e.g., "sucrose" and "glucose") across three embedding versions using cosine similarity and Euclidean distance.
- Added three images visualizing word distances, which help understand how embeddings build relationships between words and reveal which model best captures semantic similarity.

## Visual Comparisons
See the following images for visual comparison of word distances and relationships in different embedding versions and open-source models:
- ![Embedding Comparison 1](/v1_version_embedding.png)
- ![Embedding Comparison 2](/v2_version_embedding.png)
- ![Embedding Comparison 3](/v3_version_embedding.png)
- ![GloVe Embedding Comparison](/glove_open_source_word_embedding.png)
- ![BERT Embedding Comparison](/bert_embedding.png)


## Insights from Open Source Models
- The open-source GloVe embedding is able to capture more meaningful relationships, with similar words clustered together much better than the toy versions.
- BERT is a context-dependent embedding model: the similarity score for a word pair can vary depending on the sentence or context in which the words appear.
- These results highlight the advantage of large-scale, context-aware models for capturing semantic meaning and relationships in language.

## Next Steps

---
*This project demonstrates the basics of word embedding training and evaluation using a toy dataset. For best results, use a larger and more diverse corpus.*