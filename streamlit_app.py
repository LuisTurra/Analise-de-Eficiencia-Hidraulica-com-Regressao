import streamlit as st
import json
import os
from datetime import datetime

st.set_page_config(page_title="Resultados do Treinamento", layout="wide")
st.title("📊 Resultados dos Treinamentos - Eficiência Hidráulica")



# Função para carregar JSON com fallback
def load_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Carregar os dois
completo = load_results('training_results_completo.json')
fisico = load_results('training_results_fisico.json')

if completo is None and fisico is None:
    st.error("Nenhum arquivo de resultados encontrado. Rode os scripts de treinamento primeiro.")
    st.stop()

col1, col2 = st.columns(2)

# Coluna Esquerda: Completo
with col1:
    if completo:
        st.subheader("Modelo Completo (todos sensores)")
        st.caption(f"Gerado em: {completo.get('timestamp', '—')}")
        
        st.metric("MSE Linear", f"{completo['mse_linear']:.4f}")
        st.metric("MSE Ridge", f"{completo['mse_ridge']:.4f}")
        st.metric("MSE Lasso", f"{completo['mse_lasso']:.4f}")
        st.success(f"**Melhor: {completo['best_model']}** (MSE = {completo['best_mse']:.4f})")
        
        with st.expander("Detalhes"):
            st.write(f"Features: {completo['num_features']}")
            st.write(f"Ciclos: {completo['num_cycles']}")
            st.write(f"Split: {completo['split']}")
            st.write(f"Épocas: {completo['epochs']}")
    else:
        st.info("Resultados do modelo completo não encontrados.")

# Coluna Direita: Físico
with col2:
    if fisico:
        st.subheader("Modelo Físico (sem CE/CP)")
        st.caption(f"Gerado em: {fisico.get('timestamp', '—')}")
        
        st.metric("MSE Linear", f"{fisico['mse_linear']:.4f}")
        st.metric("MSE Ridge", f"{fisico['mse_ridge']:.4f}")
        st.metric("MSE Lasso", f"{fisico['mse_lasso']:.4f}")
        st.success(f"**Melhor: {fisico['best_model']}** (MSE = {fisico['best_mse']:.4f})")
        
        with st.expander("Detalhes"):
            st.write(f"Features: {fisico['num_features']}")
            st.write(f"Ciclos: {fisico['num_cycles']}")
            st.write(f"Split: {fisico['split']}")
            st.write(f"Épocas: {fisico['epochs']}")
    else:
        st.info("Resultados do modelo físico não encontrados.")

# Gráficos (se existirem)
st.markdown("---")
st.subheader("Gráficos Gerados")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.caption("Modelo Completo")
    if os.path.exists("comparison.png"):
        st.image("comparison.png",use_container_width=True

)
    if os.path.exists("3d_plot.png"):
        st.image("3d_plot.png", use_container_width=True

)

with col_g2:
    st.caption("Modelo Físico")
    if os.path.exists("comparison_physical.png"):
        st.image("comparison_physical.png", use_container_width=True

)
    if os.path.exists("3d_plot_physical.png"):
        st.image("3d_plot_physical.png",use_container_width=True

)

st.caption(f"Dashboard atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M')}")