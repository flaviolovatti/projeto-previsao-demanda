# projeto-previsao-demanda

Projeto simples de previsão de demanda utilizando técnicas clássicas de séries temporais.

## O que o código faz

Este projeto carrega um dataset de demanda de produtos, realiza um tratamento inicial dos dados, seleciona uma categoria de produtos para análise, agrega a demanda em séries semanais e aplica dois modelos de previsão:

- Regressão linear com features de tendência e sazonalidade
- Modelo Holt-Winters (ETS)

O fluxo também gera métricas de erro para comparar os modelos e salva gráficos e arquivos de saída em uma pasta dedicada.

## Estrutura do repositório

- `src/cpd_aula14.py`: ponto de entrada do pipeline
- `src/forecasting_pipeline.py`: implementação do fluxo de carga, limpeza, modelagem e avaliação
- `outputs/`: gráficos e arquivos gerados pela execução

## Como executar

1. Instale as dependências:
   ```bash
   pip install numpy pandas plotnine statsmodels scikit-learn
   ```

2. Execute o pipeline:
   ```bash
   python src/cpd_aula14.py
   ```

## Observações

- O código foi refatorado para ficar mais organizado e próximo de boas práticas de estruturação, mantendo a reprodutibilidade do fluxo original.
- Os resultados podem variar conforme a versão das bibliotecas e o ambiente de execução.
