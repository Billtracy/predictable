"""Standalone NBA model.

Fully separate from the football pipelines: its own database (nba.db), ratings and
evaluation code. Nothing here imports models/, predictors/, db/ or ucl/, and nothing
writes to predictable.db or ucl.db.
"""
