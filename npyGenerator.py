import pandas as pd
import numpy as np

# Defina o caminho do arquivo Excel
caminho_arquivo_excel = "data.xlsx"

# Leia os dados do Excel usando pandas
dados_excel = pd.read_excel(caminho_arquivo_excel, header=None)

# Converta os dados para o formato do tipo numpy array
dados_numpy = dados_excel.to_numpy()

# Salve os dados em um arquivo com extensão .npy
np.save("dados_preprocessados.npy", dados_numpy)
