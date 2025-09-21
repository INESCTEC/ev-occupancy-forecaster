# PlugPredict: Occupancy Forecasting with Logistic Regression

**PlugPredict** is a Python tool for forecasting occupancy in EV charging stations (or similar plug-in resources).  
It uses historical binary occupancy data (`0 = free`, `1 = occupied`) stored in `.txt` files and trains a logistic regression model (NumPy-only, no scikit-learn).  
Predictions are generated for the next **12 hours** at **5-minute intervals**, and results are saved as `.json` files or served through an API.

---

## Features

- Logistic regression implemented from scratch with NumPy  
- Reads historical occupancy data from `.txt` logs  
- Trains on all available history  
- Forecasts 12 hours ahead (144 × 5-minute steps)  
- Outputs results as `.json` (timestamp + predicted state)  
- Configurable using environment variables or through an API  


---


## Workflow

<p align="center">
  <img src="docs/flowchart.png" alt="Voltune Flowchart" width="600"/>
</p>
---

## File Structure

```
PlugPredict/
├── PlugPredict.py        # Main script for CLI usage
├── api.py                # FastAPI wrapper for HTTP usage
├── README.md             # Documentation
├── database/             # Folder with historical .txt files
└── json/                 # Folder where predictions are saved
```

---

## Input File Format

### TXT (Historical Occupancy Data)
- Format:  
  ```
  YYYY-MM-DD HH:MM:SS 	 0_or_1
  ```
- Example:
  ```
  2025-09-18 08:00:00    0
  2025-09-18 08:05:00    1
  2025-09-18 08:10:00    1
  ```
- Must be continuous in **5-minute resolution**. Missing timestamps are automatically filled with `0`.

### JSON (Forecast Output)
After running the script or API, predictions are returned/saved as JSON:

```json
[
  {"timestamp": "2025-09-19 08:05:00", "value": 1},
  {"timestamp": "2025-09-19 08:10:00", "value": 0}
]
```

---

## Usage: Command Line

PlugPredict can be used directly from the command line with environment variables.

### Environment Variables

- `INPUT_FOLDER` → Path to folder containing `.txt` history files  
- `OUTPUT_FOLDER` → Path to folder where `.json` predictions will be written  

### Example

#### PowerShell (Windows)
```powershell
$env:INPUT_FOLDER="C:/Users/You/database"
$env:OUTPUT_FOLDER="C:/Users/You/json"
python PlugPredict.py
```

#### Bash (Linux/macOS)
```bash
export INPUT_FOLDER="/home/you/database"
export OUTPUT_FOLDER="/home/you/json"
python PlugPredict.py
```

If variables are missing, the script will exit with an error message.

---

## Usage: API

PlugPredict also provides a REST API implemented with **FastAPI**.

### Run the API

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 5000
```

The interactive documentation will be available at:  
[http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs)

### Endpoint

**POST** `/forecast`  
Upload a `.txt` file with occupancy history and get a 12-hour forecast.

#### Query Parameters
- `threshold` (float, default=0.6) → Probability threshold for classification

#### Request Example (curl)

```bash
curl -X POST "http://127.0.0.1:5000/forecast?threshold=0.6"   -F "file=@./database/station1.txt"
```

#### Response Example

```json
[
  {"timestamp": "2025-09-21 10:05:00", "value": 0},
  {"timestamp": "2025-09-21 10:10:00", "value": 1},
  {"timestamp": "2025-09-21 10:15:00", "value": 1}
]
```

---

## How It Works

1. Loads all `.txt` occupancy logs from `INPUT_FOLDER` or uploaded via API  
2. Converts timestamps into **cyclical features**: hour, minute, day of week, weekend indicator  
3. Trains a logistic regression model (gradient descent + optional L2 regularization)  
4. Generates predictions for the next **12 hours** at **5-minute intervals**  
5. Outputs results as `.json` or API response  

---

## Requirements

- Python 3.7+  
- NumPy  
- Pandas  
- FastAPI (for API mode)  
- Uvicorn (for API server)

Install dependencies:
```bash
pip install numpy pandas fastapi uvicorn
```

---

## Example Output

```
[OK] Saved forecast to: json/station1_pred.json
```

JSON file will look like:
```json
[
  {"timestamp": "2025-09-19 08:05:00", "value": 1},
  {"timestamp": "2025-09-19 08:10:00", "value": 0},
  {"timestamp": "2025-09-19 08:15:00", "value": 0}
]
```

---

## Customization

- Change forecast horizon (default = 12h) by modifying `periods=144` in the code  
- Adjust probability threshold (default = 0.6) for binary classification  
- Add new time-based features for richer predictions  

---

## Authors

- José P. Sousa (jose.p.sousa@inesctec.pt)  
- Gil Sampaio (gil.s.sampaio@inesctec.pt)  
