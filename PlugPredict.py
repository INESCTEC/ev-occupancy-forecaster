import os
import json
import numpy as np
import pandas as pd

# -----------------------
# Logistic Regression (NumPy only)
# -----------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def add_bias(X: np.ndarray) -> np.ndarray:
    """Add a bias column (1s) to the input features."""
    return np.c_[np.ones((X.shape[0], 1)), X]

def predict_logistic_regression(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict probabilities using logistic regression."""
    return sigmoid(X @ w)

def logistic_regression_train(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    iterations: int = 3000,
    l2: float = 0.0,
) -> np.ndarray:
    """Train logistic regression using gradient descent (with optional L2 regularization)."""
    Xb = add_bias(X)
    w = np.zeros(Xb.shape[1], dtype=float)
    n = y.size

    for _ in range(iterations):
        y_pred = sigmoid(Xb @ w)
        gradient = (Xb.T @ (y_pred - y)) / n

        # Apply L2 regularization (excluding bias)
        if l2 > 0:
            regularization = np.r_[0.0, w[1:]]
            gradient += l2 * regularization

        w -= learning_rate * gradient

        # Early stopping
        if np.linalg.norm(gradient) < 1e-6:
            break

    return w

# -----------------------
# Data Loading
# -----------------------

def load_txt_to_dataframe(filepath: str) -> pd.DataFrame:
    """Load occupancy data from a TXT file (timestamp \\t 0/1)."""
    df = pd.read_csv(filepath, sep='\t', header=None, names=['timestamp', 'occupied'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['occupied'] = df['occupied'].astype(int)
    df = df.sort_values('timestamp')

    # Ensure 5-minute frequency, fill missing intervals with 0
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('5min').fillna(0).astype(int)
    df.reset_index(inplace=True)
    return df

# -----------------------
# Forecasting Function
# -----------------------

def forecast_12h_from_txt(filepath_txt: str, output_folder: str):
    """Train logistic regression and predict the next 12 hours (5-min resolution)."""
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filepath_txt))[0]
    output_filename = f"{base_name}_pred.json"

    # Load and process data
    df = load_txt_to_dataframe(filepath_txt)

    # Time features
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

    features = [
        'hour_sin', 'hour_cos',
        'minute_sin', 'minute_cos',
        'dow_sin', 'dow_cos',
        'is_weekend'
    ]

    X_train = df[features].to_numpy(dtype=float)
    y_train = df['occupied'].to_numpy(dtype=int)

    # Future 12h timestamps (144 intervals of 5min)
    last_timestamp = df['timestamp'].max()
    future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=5), periods=144, freq='5min')
    df_future = pd.DataFrame({'timestamp': future_timestamps})

    # Future features
    df_future['hour'] = df_future['timestamp'].dt.hour
    df_future['minute'] = df_future['timestamp'].dt.minute
    df_future['dayofweek'] = df_future['timestamp'].dt.dayofweek
    df_future['is_weekend'] = (df_future['dayofweek'] >= 5).astype(int)
    df_future['hour_sin'] = np.sin(2 * np.pi * df_future['hour'] / 24)
    df_future['hour_cos'] = np.cos(2 * np.pi * df_future['hour'] / 24)
    df_future['minute_sin'] = np.sin(2 * np.pi * df_future['minute'] / 60)
    df_future['minute_cos'] = np.cos(2 * np.pi * df_future['minute'] / 60)
    df_future['dow_sin'] = np.sin(2 * np.pi * df_future['dayofweek'] / 7)
    df_future['dow_cos'] = np.cos(2 * np.pi * df_future['dayofweek'] / 7)

    X_future = df_future[features].to_numpy(dtype=float)

    # Train and predict
    model = logistic_regression_train(X_train, y_train, learning_rate=0.05, iterations=3000, l2=0.01)
    y_prob = predict_logistic_regression(add_bias(X_future), model)
    y_pred = (y_prob >= 0.6).astype(int)

    predictions = [
        {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "value": int(val)}
        for ts, val in zip(df_future['timestamp'], y_pred)
    ]

    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved forecast to: {output_path}")

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    input_folder = os.environ.get("INPUT_FOLDER")
    output_folder = os.environ.get("OUTPUT_FOLDER")

    if not input_folder or not output_folder:
        print("Error: Please set the environment variables INPUT_FOLDER and OUTPUT_FOLDER.")
        print("Example (PowerShell):")
        print("  $env:INPUT_FOLDER='C:/path/to/input_folder'")
        print("  $env:OUTPUT_FOLDER='C:/path/to/output_folder'")
        exit(1)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            full_path = os.path.join(input_folder, filename)
            forecast_12h_from_txt(full_path, output_folder)
