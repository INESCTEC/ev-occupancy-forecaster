from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
import numpy as np
import pandas as pd
import io
from PlugPredict import load_txt_to_dataframe, forecast_12h_from_txt

app = FastAPI(
    title="PlugPredict API",
    description="API for computing 12h plug occupancy forecast from history files.",
    version="1.0.0"
)

# Response model
class ForecastItem(BaseModel):
    timestamp: str
    value: int

@app.post(
    "/forecast",
    response_model=list[ForecastItem],
    summary="Compute 12h plug occupancy forecast",
    description=(
        "Upload a `.txt` file with historical occupancy data (timestamp TAB 0/1).\n\n"
        "The model is trained on the history and predicts the next 12 hours at 5-minute resolution."
    ),
    tags=["PlugPredict"]
)
async def compute_forecast(
    file: UploadFile = File(..., description="History file [.txt]"),
    threshold: float = Query(0.6, description="Decision threshold for occupancy (default=0.6)")
):
    # Save uploaded file temporarily
    temp_path = "temp_input.txt"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Run pipeline
    output_folder = "temp_out"
    import os, json
    os.makedirs(output_folder, exist_ok=True)
    forecast_12h_from_txt(temp_path, output_folder)

    # Build correct file path (matches PlugPredict naming)
    base_name = os.path.splitext(os.path.basename(temp_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}_pred.json")

    # Read JSON and return
    with open(output_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    return [ForecastItem(**p) for p in predictions]
