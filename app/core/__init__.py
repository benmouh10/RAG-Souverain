"""Couche cœur de l'application RAG Souverain.

Regroupe les composants transverses et le moteur RAG principal
(``RAGEngine``), consommés par l'interface Streamlit, l'évaluation RAGAS
et les pages de gestion. Issue du remaniement C2 : le moteur RAG a été
extrait de ``app.evaluation.eval_pipeline`` pour séparer les responsabilités
(évaluation ≠ moteur de production).
"""
