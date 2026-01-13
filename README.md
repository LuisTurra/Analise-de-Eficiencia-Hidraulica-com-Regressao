# Análise de Eficiência Hidráulica com Regressão 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B)](https://streamlit.io/)
[![UCI Dataset](https://img.shields.io/badge/Dataset-UCI%20Hydraulic%20Systems-green)](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)

Projeto de **análise de eficiência hidráulica** usando regressão linear  (Linear, Ridge e Lasso) no dataset UCI "Condition Monitoring of Hydraulic Systems".  
O objetivo é prever a eficiência (%) de sistemas hidráulicos a partir de dados de sensores, comparando dois cenários:  
- **Modelo Completo** (todos os sensores, incluindo virtuais CE/CP)  
- **Modelo Físico** (apenas sensores físicos, sem CE/CP)

## Visão Geral

- **Dataset**: UCI Condition Monitoring of Hydraulic Systems (2205 ciclos de 60s, 17 sensores)
- **Tarefa**: Regressão para prever Efficiency factor (SE %)
- **Features**: Estatísticas agregadas (mean, std, min, max) de cada sensor por ciclo
- **Modelos**: Linear Regression, Ridge (L2), Lasso (L1) implementados com PyTorch
- **Avaliação**: MSE no conjunto de teste (80/20 split)
- **Visualizações**: Gráficos Predicted vs Actual + 3D (Pressão × Fluxo × Eficiência)

## Diferença entre os Modelos

| Aspecto                  | Modelo Completo                  | Modelo Físico                       |
|--------------------------|----------------------------------|-------------------------------------|
| Sensores usados          | Todos (16: físicos + CE/CP)      | Apenas físicos (14)                 |
| Número de features       | 64 (16 × 4 stats)                | 56 (14 × 4 stats)                   |
| MSE típico (test set)    | ~1.4–1.6 (muito baixo)           | ~1.05-1.12 (mais baixo)             |
| Precisão                 | Excelente (quase perfeita)       | Moderada (mais realista)            |
| Aplicação real           | Menos prático (CE/CP são virtuais) | Mais realista para manutenção preditiva |


## Dashboard com Resultados do Treinamento : https://luisturra-analise-de-eficiencia-hidraulica-streamlit-app-ont94j.streamlit.app/ 

## Como Rodar

### 1. Instalar dependências

```bash
pip install pandas numpy torch scikit-learn matplotlib streamlit

## Baixe o Dataset
- ** https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems **

## Treine os Modelos

** Modelo completo (todos sensores)**
python train_and_evaluate.py

**Modelo físico (sem CE/CP)**
python train_physical_sensors.py

## Abrir Dashboard

streamlit run streamlit_resultados.py