# AI Requirement Mapper

A small NLP-based tool that maps client requirements to the most relevant system capabilities.

I built this project to explore how sentence embeddings can be used for a practical implementation problem: given a list of requirements and a list of available system capabilities, find the closest match and flag requirements that may need manual review.

## What it does

The project takes two CSV files:

- `requirements.csv` - requirements provided by a client
- `capabilities.csv` - capabilities supported by a system

It converts both into sentence embeddings using a Sentence Transformer model and compares them using cosine similarity.

For every requirement, the tool finds the closest capability and assigns a status based on the similarity score.

```text
Client Requirement
        ↓
Sentence Transformer
        ↓
Requirement Embedding
        ↓
Cosine Similarity
        ↓
Best Matching Capability
        ↓
Similarity Score
        ↓
MATCH / REVIEW