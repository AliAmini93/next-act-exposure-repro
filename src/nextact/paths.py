"""Locations used by every module. Set them with environment variables before running:

  NEXTACT_DATA     directory holding the corpus downloads at the relative paths listed in nextact.corpora.SOURCES
                   (default: ./work/data)
  NEXTACT_PRIVATE  row-level outputs: predictions, per-query scores, plans, embeddings (default: ./work/private).
                   These contain corpus-derived records; never publish them.
  NEXTACT_RESULTS  aggregate outputs: estimates with intervals, decisions, fingerprints (default: ./work/results)
  NEXTACT_WORK     parent of the defaults above and of logs/ (default: ./work)
"""
import os
from pathlib import Path

WORK = Path(os.environ.get('NEXTACT_WORK', 'work')).resolve()
DATA = Path(os.environ.get('NEXTACT_DATA', WORK / 'data')).resolve()
PRIVATE = Path(os.environ.get('NEXTACT_PRIVATE', WORK / 'private')).resolve()
RESULTS = Path(os.environ.get('NEXTACT_RESULTS', WORK / 'results')).resolve()
LOGS = WORK / 'logs'
PACKAGE = Path(__file__).resolve().parent
