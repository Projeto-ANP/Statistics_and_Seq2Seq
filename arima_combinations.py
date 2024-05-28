from all_functions import *
import os
import pickle
from sklearn.ensemble import VotingRegressor
from VotingCombination import VotingCombination
from aeon.visualisation import plot_series
pastas = [
        #   './Statistics_and_Seq2Seq/results/arima/rolling/gasolinac',
          './results/arima/rolling/oleodiesel',
        #   './Statistics_and_Seq2Seq/results/arima/rolling/etanolhidratado',
        #   './Statistics_and_Seq2Seq/results/arima/rolling/gasolinadeaviacao',
        #   './Statistics_and_Seq2Seq/results/arima/rolling/glp',
        #   './Statistics_and_Seq2Seq/results/arima/rolling/oleocombustivel',
        #   './Statistics_and_Seq2Seq/results/arima/rolling/querosenedeaviacao',
        #   './Statistics_and_Seq2Seq/results/arima/rolling/queroseneiluminante'
          ]


resultado_final_por_uf = {}

for pasta in pastas:
    if os.path.isdir(pasta):
        for arquivo in os.listdir(pasta):
            if arquivo.endswith('.csv'):
                caminho_arquivo = os.path.join(pasta, arquivo)
                df = pd.read_csv(caminho_arquivo, delimiter=';')
                
                df_filtrado = df[df['DATA'].isin(['normal', 'deseasonal', 'log']) & df['SAVED MODEL'].notna() & df['UF'].notna()]
                for indice, linha in df_filtrado.iterrows():
                    caminho_pickle = linha['SAVED MODEL']
                    
                    # with open(caminho_pickle, 'rb') as f:
                    #     arima_model = pickle.load(f)
                    predictions = [linha[f'P{i}'] for i in range(1, 13)]
                    uf = linha['UF']
                    if uf not in resultado_final_por_uf:
                        resultado_final_por_uf[uf] = []
                    resultado_final_por_uf[uf].append((linha['DATA'], predictions))

uf = 'PI'
derivado = "oleodiesel"
horizon = 12

df = pd.read_csv(f"../datasets/venda/mensal/uf/{derivado}/mensal_{uf.lower()}_{derivado}.csv", header=0, parse_dates=['timestamp'], sep=";", date_parser=custom_parser)
df['timestamp']=pd.to_datetime(df['timestamp'],infer_datetime_format=True)
df = df.set_index('timestamp',inplace=False)
df.index = df.index.to_period('M')
series = df['m3']
all_series_test = []
train, test = train_test_stats(series, horizon)

voting = VotingCombination(resultado_final_por_uf[uf], combination='mean')
predictions = voting.predict()
print(predictions)
plot_series(test, predictions, labels=["Test", "Predictions"])