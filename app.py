# CÉLULA 1: Importações
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os
import gdown

print("Bibliotecas importadas.")

# CÉLULA 2: Download dos modelos do Google Drive
print("Baixando modelos...")

# Dicionário com IDs dos arquivos e nomes de saída
# ATENÇÃO: Substitua pelos seus IDs de arquivo do Google Drive
files_to_download = {
    'SEU_ID_DO_SCALER_AQUI': 'scaler.joblib',
    'SEU_ID_DO_MODELO_RFR_AQUI': 'modelo_rfr.joblib',
    'SEU_ID_DO_MODELO_GBR_AQUI': 'modelo_gbr.joblib'
}

# Cria a pasta 'models' se não existir
os.makedirs('models', exist_ok=True)

# Baixa cada arquivo
for file_id, output_name in files_to_download.items():
    output_path = os.path.join('models', output_name)
    if not os.path.exists(output_path):
        print(f"Baixando {output_name}...")
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
    else:
        print(f"{output_name} já existe.")
print("Downloads concluídos.")

# CÉLULA 3: Carregamento dos modelos
print("Carregando modelos...")
scaler = joblib.load('models/scaler.joblib')
modelo_rfr = joblib.load('models/modelo_rfr.joblib')
modelo_gbr = joblib.load('models/modelo_gbr.joblib')
print("Modelos carregados.")


# CÉLULA 4: Função de Predição e Classificação (CORRIGIDA)
def predict_and_classify(pot, tia, glu):
    # Criar um DataFrame com os dados de entrada
    dados = pd.DataFrame({'POT': [pot], 'TIA': [tia], 'GLU': [glu]})
    
    # Padronizar os dados de entrada
    dados_padronizados = scaler.transform(dados)
    
    # Fazer previsões com os dois modelos
    pred_rfr = modelo_rfr.predict(dados_padronizados)
    pred_gbr = modelo_gbr.predict(dados_padronizados)
    
    # Calcular a média das previsões (ensemble)
    pred_final = (pred_rfr + pred_gbr) / 2
    valor_predito = pred_final[0] # Pega o valor numérico da predição

    # Lógica de classificação baseada no valor predito
    if valor_predito < 70:
        resultado = f"Hipoglicemia (Valor: {valor_predito:.2f})"
    elif 70 <= valor_predito <= 99:
        resultado = f"Normal (Valor: {valor_predito:.2f})"
    elif 100 <= valor_predito <= 125:
        resultado = f"Pré-diabetes (Valor: {valor_predito:.2f})"
    else:
        resultado = f"Diabetes (Valor: {valor_predito:.2f})"
        
    return resultado

# CÉLULA 5: Interface do Gradio (CORRIGIDA)
iface = gr.Interface(
    fn=predict_and_classify, # Usa a nova função
    inputs=[
        gr.Number(label="POT"),
        gr.Number(label="TIA"),
        gr.Number(label="GLU")
    ],
    outputs=gr.Textbox(label="Resultado da Glicemia"), # A saída agora é um texto
    title="Deploy do Modelo Ensemble para Sensor de Glicose",
    description="Insira os valores de POT, TIA e GLU para obter a previsão e a classificação do nível de glicemia."
)

# Lançar a aplicação
iface.launch()
