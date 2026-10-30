"""Same-dialogue exposure in next-act evaluation: code that produces the reported estimates.

Modules: corpora (SGD, STAR, AMI readers), protocol (randomised presence design and estimators), lexical, encoder,
knn and lexical_explore (randomised presence experiments), icl, icl_explore, icl_fresh (retrieval in-context learning,
first and fresh samples), unlabelled (unlabelled-context control), consistency (separately written recomputations and
common support), window_parts and window_followup (follow-up decomposition), dense (dense-retriever prevalence),
exposure (protocol-implied exposure arithmetic), report (Markdown summaries), settings (prespecified constants) and
checks (independent recomputations). Paths are configured in nextact.paths.
"""
__version__ = '0.1.0'
