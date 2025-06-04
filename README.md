# UC Capital Hybrid Forecast Pipeline

## Overview
This repo provides a modular, production-ready pipeline for hybrid time series forecasting using SARIMA+XGBoost and LSTM models. It supports GPU acceleration, Google Colab, and robust walk-forward backtesting.

- **SARIMA-XGB**: Monthly hybrid (SARIMA + XGBoost-on-residuals) pipeline
- **LSTM**: Daily technical LSTM pipeline
- **Walk-forward backtest**: Comparable metrics for both models
- **Colab-ready**: One-click setup and Google Drive integration
- **GPU support**: XGBoost with CUDA for massive speedup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your_notebook_link_here)

## Usage

```bash
python main.py --in_file data.xlsx --out_dir results --model {sarima_xgb,lstm} --gpu --colab --log INFO
```

- `--model {sarima_xgb,lstm}`: Choose model pipeline
- `--gpu`: Enable GPU for XGBoost if available
- `--colab`: Mount Google Drive and auto-install requirements
- `--in_file`, `--out_dir`: Input/output paths
- `--log`: Logging level (DEBUG, INFO, etc.)

## Environment
- `requirements.txt` for pip/Colab
- `environment.yml` (CPU) and `environment-gpu.yml` (CUDA 11.x) for conda
- For Colab GPU: `pip install xgboost-cu116`

## GPU vs CPU Timing Example
| Model         | CPU (6h) | GPU (45min) |
|--------------|----------|-------------|
| SARIMA-XGB   | 6h       | 45min       |
| LSTM         | 4h       | 30min       |

## Tests
- Run `pytest tests/` to check pipeline on a 1-month sample CSV

## Security
- `.gitignore` excludes all generated outputs, notebooks, and large files

## Experiment Tracking (optional)
- Add `wandb.init(project="uc-sample")` or TensorBoard for experiment tracking


![2008](https://github.com/user-attachments/assets/00dcd326-2325-4176-a392-034127b08e34)
![2018](https://github.com/user-attachments/assets/fbdc5797-657b-46fe-8791-2b9b06174a3a)
![2020](https://github.com/user-attachments/assets/e0c2f929-4596-45b4-b38f-7964299583b2)


