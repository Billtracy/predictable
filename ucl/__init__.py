"""Standalone Champions League model.

Fully separate from the EPL pipeline: its own database (ucl.db), its own ratings
and simulation code. Nothing here imports models/, predictors/ or db/, and nothing
writes to predictable.db, so the Prem model and its baselines are untouched.
"""
