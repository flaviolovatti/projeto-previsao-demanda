# -*- coding: utf-8 -*-
"""Ponto de entrada para o pipeline de previsão de demanda.

Este módulo mantém o fluxo original, mas delega a lógica para um módulo
mais organizado e reutilizável em src/forecasting_pipeline.py.
"""

from forecasting_pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline()
