import argparse
import logging
import sys
import os
import pandas as pd
from data_utils import _to_num, _parse_cols
from models import build_sarima_xgb, build_lstm
from backtest import walk_forward_hybrid, walk_forward_lstm
import xgboost as xgb

def setup_logging(log_level, out_dir):
    log_path = os.path.join(out_dir, 'run.log')
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    )

def detect_cuda():
    try:
        return xgb.core._has_cuda_support()
    except Exception:
        return False

def mount_colab():
    from google.colab import drive
    drive.mount('/content/drive')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='input.xlsx')
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--model', type=str, choices=['sarima_xgb', 'lstm'], default='sarima_xgb')
    parser.add_argument('--log', type=str, default='INFO')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(args.log, args.out_dir)

    if args.colab:
        mount_colab()
        os.system('pip install -r requirements.txt')

    use_cuda = args.gpu and detect_cuda()
    logging.info(f'CUDA enabled: {use_cuda}')

    if args.model == 'sarima_xgb':
        logging.info('Loading input file...')
        raw = pd.read_excel(args.in_file, header=0, skiprows=2)
        params = dict(
            win_size=120,
            step_size_ro=5,
            win_wf=241,
            step_wf=1,
            lag_12=12,
            hit_grid=[dict(max_depth=d, eta=e) for d in [2,3,4] for e in [0.05,0.1,0.2]],
            xgb_base=dict(
                max_depth=3, eta=0.1, tree_method="hist", predictor="cpu_predictor",
                subsample=1, colsample_bytree=1, seed=123, deterministic_histogram=1,
                objective="reg:squarederror", verbosity=0
            ),
            seed=123,
            use_cuda=use_cuda
        )
        logging.info('Building SARIMA-XGB hybrid model...')
        result = build_sarima_xgb(raw, **params)
        logging.info('Running walk-forward backtest...')
        wf0 = walk_forward_hybrid(result['ret'], result['X0'], 'lag0', result['fit0'], result['xgb0'], params['win_wf'], params['step_wf'], params['lag_12'])
        wf12 = walk_forward_hybrid(result['ret'], result['X12'], 'lag12', result['fit12'], result['xgb12'], params['win_wf'], params['step_wf'], params['lag_12'])
        wfAR = walk_forward_hybrid(result['ret'], result['XAR'], 'arimax', result['fitAR'], result['xgbAR'], params['win_wf'], params['step_wf'], params['lag_12'])
        wf0.to_excel(os.path.join(args.out_dir, 'WF_Lag0.xlsx'), index=False)
        wf12.to_excel(os.path.join(args.out_dir, 'WF_Lag12.xlsx'), index=False)
        wfAR.to_excel(os.path.join(args.out_dir, 'WF_ARIMAX.xlsx'), index=False)
        logging.info('Results saved to output directory.')
    elif args.model == 'lstm':
        logging.info('Loading input file...')
        raw = pd.read_excel(args.in_file, header=None, skiprows=2)
        params = dict(
            cand_train=[252, 504, 756, 1260],
            cand_seq=[20, 40, 60],
            step_wf=1,
            early_stop=20,
            max_lev=2.0,
            stop_pct=-5.0,
            seed=123
        )
        logging.info('Building LSTM model and running walk-forward grid search...')
        wf, stats = build_lstm(raw, **params)
        if wf is not None:
            wf.to_excel(os.path.join(args.out_dir, 'WF_LSTM.xlsx'), index=False)
            logging.info('LSTM walk-forward log saved to output directory.')
        else:
            logging.error('No valid parameter set found for LSTM.')

if __name__ == '__main__':
    main() 