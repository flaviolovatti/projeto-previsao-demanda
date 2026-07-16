# -*- coding: utf-8 -*-
"""Pipeline simples para preprocessamento, exploração e previsão de demanda."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotnine as p9
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

DATASET_URL = "https://aluno.analisemacro.com.br/download/69280/?tmstv=1768230842"
START_DATE = pd.Timestamp("2012-01-01")
CUTOFF_DATE = pd.Timestamp("2015-01-01")
END_DATE = pd.Timestamp("2017-01-01")
SEASONAL_PERIOD = 52
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def configure_environment() -> None:
    """Configurações básicas para manter a saída mais legível."""
    pd.options.display.float_format = "{:.2f}".format


def load_raw_data(url: str = DATASET_URL) -> pd.DataFrame:
    """Carrega os dados brutos do dataset."""
    return pd.read_csv(url, compression="zip")


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Realiza o tratamento inicial das colunas do dataset."""
    return (
        raw_df.copy()
        .assign(
            Date=lambda x: pd.to_datetime(x["Date"], format="%Y/%m/%d"),
            Order_Demand=lambda x: (
                x["Order_Demand"]
                .astype(str)
                .str.replace(r"\(", "", regex=True)
                .str.replace(r"\)", "", regex=True)
                .astype(int)
            ),
        )
    )


def build_weekly_series(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Agrupa a demanda por semana para a categoria selecionada."""
    return (
        df.query("Product_Category == @category")
        .groupby("Date")["Order_Demand"]
        .sum()
        .to_frame(name="Order_Demand")
        .loc[lambda x: x.index >= START_DATE]
        .resample("W")
        .sum()
        .loc[lambda x: x.index <= END_DATE]
    )


def create_regression_features(series: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis explicativas para a regressão linear."""
    regression_df = series.copy()
    regression_df["tendencia"] = np.arange(1, len(regression_df) + 1) + regression_df["Order_Demand"].mean()
    regression_df["sazonalidade"] = np.sin(2 * np.pi * regression_df.index.month / 12)
    return regression_df


def split_train_test(regression_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa os dados em treino e teste."""
    train_df = regression_df.loc[regression_df.index <= cutoff_date].copy()
    test_df = regression_df.loc[regression_df.index > cutoff_date].copy()
    return train_df, test_df


def train_linear_regression(train_df: pd.DataFrame) -> Pipeline:
    """Treina um modelo de regressão linear com padronização."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(train_df[["tendencia", "sazonalidade"]], train_df["Order_Demand"])
    return model


def train_ets_model(series: pd.Series) -> object:
    """Treina um modelo Holt-Winters com sazonalidade aditiva."""
    return ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
    ).fit(optimized=True)


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas simples de erro para avaliação."""
    errors = y_true - y_pred
    return {
        "erro_medio": float(np.mean(errors)),
        "erro_medio_absoluto": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def save_outputs(df_alvo: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Salva artefatos reutilizáveis para análise posterior."""
    df_alvo.to_csv(OUTPUT_DIR / "serie_alvo.csv", index=True)
    predictions.to_csv(OUTPUT_DIR / "previsoes.csv", index=True)

    temporal_plot = (
        p9.ggplot(df_alvo.reset_index())
        + p9.aes(x="Date", y="Order_Demand")
        + p9.geom_line()
    )
    temporal_plot.save(filename=str(OUTPUT_DIR / "serie_temporal.png"))

    histogram_plot = (
        p9.ggplot(df_alvo.reset_index())
        + p9.aes(x="Order_Demand")
        + p9.geom_histogram()
    )
    histogram_plot.save(filename=str(OUTPUT_DIR / "histograma.png"))


def run_pipeline() -> None:
    """Executa o fluxo completo de coleta, tratamento, modelagem e avaliação."""
    configure_environment()

    raw_df = load_raw_data()
    df_tratado = clean_data(raw_df)

    category = df_tratado["Product_Category"].value_counts().idxmax()
    df_alvo = build_weekly_series(df_tratado, category)

    print(f"Categoria selecionada: {category}")
    print(df_alvo.head())

    stl = STL(df_alvo["Order_Demand"], period=SEASONAL_PERIOD).fit()
    stl.plot()

    regression_df = create_regression_features(df_alvo)
    train_df, test_df = split_train_test(regression_df, CUTOFF_DATE)

    linear_model = train_linear_regression(train_df)
    y_prev_rl = linear_model.predict(test_df[["tendencia", "sazonalidade"]])

    ets_model = train_ets_model(train_df["Order_Demand"])
    y_prev_ets = ets_model.forecast(len(test_df))

    predictions = (
        pd.Series(y_prev_rl, index=test_df.index, name="RL")
        .to_frame()
        .join(pd.Series(y_prev_ets, index=test_df.index, name="ETS"))
        .join(test_df["Order_Demand"].rename("Order_Demand"))
    )

    print("\nMétricas RL")
    for metric_name, metric_value in calculate_metrics(predictions["Order_Demand"], predictions["RL"]).items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nMétricas ETS")
    for metric_name, metric_value in calculate_metrics(predictions["Order_Demand"], predictions["ETS"]).items():
        print(f"{metric_name}: {metric_value:.4f}")

    final_ets_model = train_ets_model(df_alvo["Order_Demand"])
    future_forecast = final_ets_model.forecast(52)
    predictions_full = df_alvo.join(future_forecast.rename("ETS"), how="outer")
    save_outputs(df_alvo, predictions_full[["Order_Demand", "ETS"]])

    print("\nPrevisão futura (52 semanas):")
    print(future_forecast.head())


if __name__ == "__main__":
    run_pipeline()
