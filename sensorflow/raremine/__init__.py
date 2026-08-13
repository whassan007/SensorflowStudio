"""Multimodal rare-event mining & perception QA for costumed pedestrians.

Core separation of powers (never collapsed into one judgment):
  - the MINER proposes candidates from available sensor evidence only;
  - AUTOMATED VALIDATION measures them against ground truth when present;
  - HUMAN validation confirms or rejects;
  - STATISTICS (curator metrics) determine how good the mining is;
  - TRAINING usage is a governed decision with an explicit leakage guard.
"""
