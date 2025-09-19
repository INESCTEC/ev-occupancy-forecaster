import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Logistic regression (NumPy only)
# -----------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones((X.shape[0], 1)), X]

def predict_logistic_regression(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)

def logistic_regression_train(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    iterations: int = 3000,
    l2: float = 0.0,
) -> np.ndarray:
    Xb = add_bias(X)
    w = np.zeros(Xb.shape[1], dtype=float)
    n = y.size

    for i in range(iterations):
        y_pred = sigmoid(Xb @ w)
        grad = (Xb.T @ (y_pred - y)) / n
        if l2 > 0:
            reg = np.r_[0.0, w[1:]]  # don't regularize bias
            grad += l2 * reg
        w -= learning_rate * grad
        if np.linalg.norm(grad) < 1e-6:
            break
    return w

# -----------------------
# Load data from TXT
# -----------------------

def carregar_txt_para_dataframe(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep='\t', header=None, names=['timestamp', 'sessao'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['sessao'] = df['sessao'].astype(int)
    df = df.sort_values('timestamp')

    # Fill 5min frequency, fill missing values as 0 (not occupied)
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('5min').fillna(0).astype(int)
    df.reset_index(inplace=True)
    return df

# -----------------------
# Forecast + Plot
# -----------------------

def previsao_12h_txt(filepath_txt: str, nome_saida: str = "saida", pasta_out: str = "json_txt"):
    os.makedirs(pasta_out, exist_ok=True)
    df = carregar_txt_para_dataframe(filepath_txt)

    # Features temporais
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    features = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos', 'is_weekend']

    # Janela temporal
    ultimo_timestamp = df['timestamp'].max()
    df_treino = df.copy()

    X_train = df_treino[features].to_numpy(dtype=float)
    y_train = df_treino['sessao'].to_numpy(dtype=int)

    # Criar timestamps futuros
    futuros_timestamps = pd.date_range(start=ultimo_timestamp + pd.Timedelta(minutes=5), periods=144, freq='5min')
    df_futuro = pd.DataFrame({'timestamp': futuros_timestamps})
    df_futuro['hour'] = df_futuro['timestamp'].dt.hour
    df_futuro['minute'] = df_futuro['timestamp'].dt.minute
    df_futuro['dayofweek'] = df_futuro['timestamp'].dt.dayofweek
    df_futuro['is_weekend'] = (df_futuro['dayofweek'] >= 5).astype(int)
    df_futuro['hour_sin'] = np.sin(2 * np.pi * df_futuro['hour'] / 24)
    df_futuro['hour_cos'] = np.cos(2 * np.pi * df_futuro['hour'] / 24)
    df_futuro['minute_sin'] = np.sin(2 * np.pi * df_futuro['minute'] / 60)
    df_futuro['minute_cos'] = np.cos(2 * np.pi * df_futuro['minute'] / 60)
    df_futuro['dow_sin'] = np.sin(2 * np.pi * df_futuro['dayofweek'] / 7)
    df_futuro['dow_cos'] = np.cos(2 * np.pi * df_futuro['dayofweek'] / 7)

    X_futuro = df_futuro[features].to_numpy(dtype=float)

    # Treinar modelo (NumPy)
    model = logistic_regression_train(X_train, y_train, learning_rate=0.05, iterations=3000, l2=0.01)
    y_prob = predict_logistic_regression(add_bias(X_futuro), model)
    y_pred = (y_prob >= 0.6).astype(int)

    # Construir JSON
    previsoes = [
        {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "value": int(val)}
        for ts, val in zip(df_futuro['timestamp'], y_pred)
    ]

    nome_json = f"previsao_{nome_saida}.json"
    caminho_out = os.path.join(pasta_out, nome_json)
    with open(caminho_out, 'w', encoding='utf-8') as f:
        json.dump(previsoes, f, indent=2, ensure_ascii=False)

    print(f"Previsão guardada: {caminho_out}")

    # --- Plotting the last 12h + predicted 12h ---
    historico_inicio = ultimo_timestamp - pd.Timedelta(hours=12)
    df_historico = df[df['timestamp'] > historico_inicio]

    df_historico_plot = df_historico[['timestamp', 'sessao']].copy()
    df_historico_plot.rename(columns={'sessao': 'value'}, inplace=True)
    df_historico_plot['type'] = 'historical'

    df_previsao_plot = pd.DataFrame(previsoes)
    df_previsao_plot['timestamp'] = pd.to_datetime(df_previsao_plot['timestamp'])
    df_previsao_plot['type'] = 'predicted'

    df_plot = pd.concat([df_historico_plot, df_previsao_plot], ignore_index=True)

    plt.figure(figsize=(16, 5))
    for tipo, grupo in df_plot.groupby('type'):
        plt.step(grupo['timestamp'], grupo['value'], where='post', label=tipo)

    plt.title(f'Ocupação - Últimas 12h + Previsão 12h ({nome_saida})')
    plt.xlabel('Tempo')
    plt.ylabel('Ocupado (1) / Livre (0)')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    nome_figura = f"plot_{nome_saida}.png"
    caminho_figura = os.path.join(pasta_out, nome_figura)
    plt.savefig(caminho_figura)
    print(f"Gráfico guardado: {caminho_figura}")


# Execução principal
if __name__ == "__main__":
    previsao_12h_txt(
        filepath_txt="database//ocupacao.txt",     # caminho para o teu ficheiro .txt
        nome_saida="carregador_01",      # nome base para os ficheiros de saída
        pasta_out="json_txt"             # pasta de saída para JSON + PNG
    )
