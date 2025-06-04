import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

def walk_forward_hybrid(y, Xbest, scenario, base_fit, xgb_model, win_wf, step_wf, lag_12):
    """Walk-forward backtest for SARIMA-XGB hybrid model."""
    if Xbest.size == 0:
        Xbest = np.empty((len(y),0))
    def _auto_sarima(y, x):
        from statsforecast.models import AutoARIMA
        mdl = AutoARIMA(season_length=12)
        mdl = mdl.fit(y, X=x if x is not None and x.size else None)
        return mdl
    def _forecast_exo_future(X, idx_t):
        if X.size == 0: return None
        fut = []
        for j in range(X.shape[1]):
            try:
                from statsforecast.models import AutoARIMA
                mdl = AutoARIMA(season_length=12)
                fit = mdl.fit(X[:idx_t, j])
                fut.append(float(fit.predict(1)[0]))
            except Exception:
                fut.append(np.nan)
        return np.array(fut).reshape(1, -1)
    def _predict_xgb(model, X_row):
        if model is None or X_row is None: return 0.0
        import xgboost as xgb
        return float(model.predict(xgb.DMatrix(X_row))[0])
    def _one_split(i):
        y_tr = y[i-win_wf:i]
        X_tr = Xbest[i-win_wf:i]
        if   scenario=="lag0":  X_te = None if X_tr.size==0 else Xbest[i:i+1]
        elif scenario=="lag12": X_te = None if X_tr.size==0 or i+1-lag_12<=0 else Xbest[i+1-lag_12:i+2-lag_12]
        else:                   X_te = None if X_tr.size==0 else _forecast_exo_future(Xbest, X_tr.shape[0])
        use_x = X_te is not None and X_tr.size
        mdl = _auto_sarima(y_tr, X_tr if use_x else None)
        arima_fc = mdl.predict(1, X=X_te)[0] if use_x else mdl.predict(1)[0]
        hyb_fc = arima_fc + _predict_xgb(xgb_model, X_te)
        act = y[i]
        return dict(SplitEnd=i-1,
                    Prediction=hyb_fc,
                    Actual=act,
                    RMSE=abs(hyb_fc-act),
                    Test_Hit=int(np.sign(hyb_fc)==np.sign(act)))
    iter_idx = range(win_wf, len(y)-1, step_wf)
    logs = Parallel(n_jobs=cpu_count(), backend="loky")(delayed(_one_split)(i) for i in tqdm(iter_idx, desc=f"WF-{scenario}", unit="split", dynamic_ncols=True))
    return pd.DataFrame(logs)

def walk_forward_lstm(df, win, seq, step_wf, early_stop, max_lev, stop_pct, seed):
    """Walk-forward backtest for LSTM model."""
    pass 