# Improving Clustering with Positive Pairs Generated from LLM-Driven Labels

1. run LPPG.py for Label-based Positive Pairs Generation
2. run small_positive_pairs.py for generating small positive pairs in Appendix **Overall Silhouette Score and Pair-Score**
3. run BYOL_train.py for using Positive Pairs to train the embedder and output the optimal check-point using **Overall Silhouette Score and Pair-Score**
4. run generate_new_labels.py for generating new labels for the final cluster
5. run estimate_K.py for generating the estimated K.