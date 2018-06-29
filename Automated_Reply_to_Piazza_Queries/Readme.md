# Automated Replies for piazza Queries:
## Requirements and References:
## Project Brief description:
There are two approaches for this problem statement, viz
* **Sentence Similarity based on Semantics nets:** 
  - Code: Automated_Reply_Similarity.pypy, based on [this](https://ieeexplore.ieee.org/document/1644735/) paper
  - This method has higher accuracy for predicting most similar sentence but is computationally more expensive
* **Finding closest similar query considering top words in vocabulary as independent vectors in N-Dimensions:**
