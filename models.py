import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA
import xgboost as xgb
from .data_utils import _to_num, _parse_cols
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import talib as ta
import optuna
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import math
import random

def build_sarima_xgb(
    raw,
    win_size,
    step_size_ro,
    win_wf,
    step_wf,
    lag_12,
    hit_grid,
    xgb_base,
    seed,
    use_cuda=False
):
    """Build SARIMA-XGB hybrid model and return fitted objects and best exogenous subsets."""
    np.random.seed(seed)
    base_feats = ["Lag1", "Lag2", "Lag3"]
    other_columns = [5,6,8,9,11,12,13,15,16,17,18]
    exo_cols = other_columns
    ret = _to_num(raw.iloc[:, 7])
    exo_df = raw.iloc[:, exo_cols].apply(_to_num)
    for lag in base_feats:
        exo_df[lag] = ret.shift(int(lag[-1]))
    exo_df = exo_df[base_feats + [c for c in exo_df.columns if c not in base_feats]]
    keep = ret.notna()
    exo_df = exo_df.fillna(method="ffill")
    ret = ret[keep].to_numpy()
    exo = exo_df[keep].to_numpy()
    idx_t = exo.shape[0]
    def _auto_sarima(y, x):
        mdl = AutoARIMA(season_length=12)
        mdl = mdl.fit(y, X=x if x is not None and x.size else None)
        return mdl
    def _forecast_exo_future(X, idx_t):
        if X.size == 0: return None
        fut = []
        for j in range(X.shape[1]):
            try:
                mdl = AutoARIMA(season_length=12)
                fit = mdl.fit(X[:idx_t, j])
                fut.append(float(fit.predict(1)[0]))
            except Exception:
                fut.append(np.nan)
        return np.array(fut).reshape(1, -1)
    def _roll_origin_metrics(y, X, scenario):
        n = len(y)
        if n <= win_size: return (np.nan, np.nan, np.nan)
        se = ae = naive_ae = hits = tot = 0
        for i in range(win_size, n - 1, step_size_ro):
            y_tr = y[i - win_size:i]
            X_tr = X[i - win_size:i]
            if   scenario == "lag0":   X_te = None if X_tr.size == 0 else X[i:i+1]
            elif scenario == "lag12":  X_te = None if X_tr.size == 0 or i + 1 - lag_12 <= 0 else X_tr[-lag_12:-lag_12+1]
            else:                      X_te = None if X_tr.size == 0 else _forecast_exo_future(X_tr, X_tr.shape[0])
            use_x = X_te is not None and X_tr.size
            try:
                mdl = _auto_sarima(y_tr, X_tr if use_x else None)
                fc  = mdl.predict(1, X=X_te)[0] if use_x else mdl.predict(1)[0]
            except Exception:
                continue
            act = y[i]
            if not np.isnan(fc) and not np.isnan(act):
                se       += (fc - act) ** 2
                ae       += abs(fc - act)
                naive_ae += abs(y_tr[-1] - act)
                hits     += int(np.sign(fc) == np.sign(act))
                tot      += 1
        if tot == 0: return (np.nan, np.nan, np.nan)
        return round(np.sqrt(se / tot), 4), round(ae / naive_ae, 4), round(hits / tot, 4)
    def _grid_search(y, X, scenario):
        n_exo = X.shape[1]
        combos = range(2 ** n_exo)
        def _eval(mask):
            cols = [i for i in range(n_exo) if mask & (1 << i)]
            X_mat = X[:, cols] if cols else np.empty((len(y), 0))
            rmse, mase, hit = _roll_origin_metrics(y, X_mat, scenario)
            subset = ",".join(map(str, cols)) if cols else "None"
            return subset, rmse, mase, hit
        res = Parallel(n_jobs=cpu_count())(
            delayed(_eval)(m) for m in tqdm(combos, desc=f"Grid-{scenario} (2^{n_exo})")
        )
        df = pd.DataFrame(res, columns=["ExoSubset", "RMSE", "MASE", "HitRate"])
        df = df.sort_values(["HitRate", "RMSE", "MASE"], ascending=[False, True, True])
        df.insert(0, "Rank", range(1, len(df) + 1))
        return df
    def _best_xgb(X, y):
        if X.shape[1] == 0: return None
        best, best_hit = None, -1
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
        for g in hit_grid:
            params = {**xgb_base, **g}
            if use_cuda:
                params['tree_method'] = 'gpu_hist'
                params['predictor'] = 'gpu_predictor'
            mdl = xgb.train(params, xgb.DMatrix(Xtr, label=ytr))
            pred = np.sign(mdl.predict(xgb.DMatrix(Xte)))
            hit  = (pred == np.sign(yte)).mean()
            if hit > best_hit: best, best_hit = mdl, hit
        return best
    df0  = _grid_search(ret, exo, "lag0")
    df12 = _grid_search(ret, exo, "lag12")
    dfAR = _grid_search(ret, exo, "arimax")
    sel0  = _parse_cols(df0.iloc[0]["ExoSubset"])
    sel12 = _parse_cols(df12.iloc[0]["ExoSubset"])
    selAR = _parse_cols(dfAR.iloc[0]["ExoSubset"])
    X0  = exo[:, sel0 ] if sel0  else np.empty((idx_t,0))
    X12 = exo[:, sel12] if sel12 else np.empty((idx_t,0))
    XAR = exo[:, selAR] if selAR else np.empty((idx_t,0))
    fit0  = _auto_sarima(ret, X0  if X0.size  else None)
    fit12 = _auto_sarima(ret, X12 if X12.size else None)
    fitAR = _auto_sarima(ret, XAR if XAR.size else None)
    res0  = ret - fit0.predict_in_sample(X=X0 if X0.size else None)
    res12 = ret - fit12.predict_in_sample(X=X12 if X12.size else None)
    resAR = ret - fitAR.predict_in_sample(X=XAR if XAR.size else None)
    xgb0  = _best_xgb(X0 , res0)  if X0.size  else None
    xgb12 = _best_xgb(X12, res12) if X12.size else None
    xgbAR = _best_xgb(XAR, resAR) if XAR.size else None
    return {
        'ret': ret,
        'exo': exo,
        'idx_t': idx_t,
        'fit0': fit0,
        'fit12': fit12,
        'fitAR': fitAR,
        'xgb0': xgb0,
        'xgb12': xgb12,
        'xgbAR': xgbAR,
        'X0': X0,
        'X12': X12,
        'XAR': XAR,
        'sel0': sel0,
        'sel12': sel12,
        'selAR': selAR,
        'df0': df0,
        'df12': df12,
        'dfAR': dfAR
    }

def build_lstm(
    raw,
    cand_train,
    cand_seq,
    step_wf,
    early_stop,
    max_lev,
    stop_pct,
    seed
):
    """Build and walk-forward an LSTM model, returning best walk-forward log DataFrame and summary stats."""
    random.seed(seed)
    np.random.seed(seed)
    def _parse_date(txt):
        import datetime as dt
        txt = str(txt).strip().strip("'").strip('"')
        for fmt in ("%y/%m/%d", "%d/%m/%y"):
            try:
                return dt.datetime.strptime(txt, fmt)
            except ValueError:
                continue
        return pd.NaT
    raw = raw.iloc[:, :6]
    raw.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    raw = raw.drop("Open", axis=1)
    raw = raw.iloc[::-1].reset_index(drop=True)
    dates = raw["Date"].apply(_parse_date)
    price = _to_num(raw["Close"])
    valid = dates.notna() & price.notna()
    dates = dates[valid]
    price = price[valid]
    high = _to_num(raw["High"])[valid]
    low = _to_num(raw["Low"])[valid]
    vol = _to_num(raw["Volume"])[valid]
    ret = price.pct_change().mul(100)
    def _tech_indicators(h, l, c, v, lb=14):
        macd, macds, _ = ta.MACD(c, 12, 26, 9)
        return pd.DataFrame({
            f"SMA_{lb}": ta.SMA(c, lb),
            f"EMA_{lb}": ta.EMA(c, lb),
            f"RSI_{lb}": ta.RSI(c, lb),
            f"ATR_{lb}": ta.ATR(h, l, c, lb) / c,
            f"ROC_{lb}": ta.ROC(c, lb),
            "MACD_hist": macd - macds,
            "OBV": ta.OBV(c, v),
            "Ret_lag1": c.pct_change().mul(100).shift(1),
        })
    def _create_seq(mat, y, seq):
        X, Y = [], []
        for i in range(seq, len(mat)):
            X.append(mat[i-seq:i]); Y.append(y[i])
        return np.asarray(X), np.asarray(Y)
    def _walk(df, win, seq):
        feats, targ = df.iloc[:,3:].values.astype(np.float32), df['Ret'].values.astype(np.float32)
        dates = df['Date'].reset_index(drop=True)
        idx = list(range(win+seq, len(df)-1, step_wf))
        scaler = StandardScaler(); logs=[]
        for i in tqdm(idx, desc=f"win={win} seq={seq}", unit="split"):
            X_raw = feats[i-win-seq:i]; y_raw=targ[i-win-seq:i]
            scaler.fit(X_raw); X_tr=scaler.transform(X_raw); X_te=scaler.transform(feats[i-seq:i])
            X_tr,y_tr = _create_seq(X_tr,y_raw,seq); X_te=X_te.reshape(1,seq,-1)
            def objective(trial):
                units = trial.suggest_int("units",16,128,log=True)
                drop  = trial.suggest_float("drop",0,0.4)
                lr    = trial.suggest_float("lr",1e-4,5e-3,log=True)
                m=keras.Sequential([
                    layers.Input((seq,X_tr.shape[2])),
                    layers.LSTM(units,dropout=drop),
                    layers.Dense(1)
                ])
                m.compile(keras.optimizers.Adam(lr),loss='mse')
                es=callbacks.EarlyStopping(patience=early_stop,restore_best_weights=True,verbose=0)
                m.fit(X_tr,y_tr,epochs=200,batch_size=32,verbose=0,callbacks=[es])
                val=m.predict(X_tr[-int(0.1*len(X_tr)):],verbose=0).flatten()
                tru=y_tr[-int(0.1*len(y_tr)):]
                return np.sqrt(((val-tru)**2).mean())
            study=optuna.create_study(direction="minimize")
            study.optimize(objective,n_trials=15,show_progress_bar=False)
            u,d,lr=study.best_params.values()
            model=keras.Sequential([
                layers.Input((seq,X_tr.shape[2])),
                layers.LSTM(u,dropout=d),
                layers.Dense(1)
            ])
            model.compile(keras.optimizers.Adam(lr),loss='mse')
            es=callbacks.EarlyStopping(patience=early_stop,restore_best_weights=True,verbose=0)
            model.fit(X_tr,y_tr,epochs=200,batch_size=32,verbose=0,callbacks=[es])
            pred=float(model.predict(X_te,verbose=0)[0,0]); actual=float(targ[i])
            side=1 if pred>=0 else -1; sigma=targ[i-win:i].std(ddof=0)
            size=min(abs(pred)/sigma if sigma>0 else 0, max_lev)
            pnl=max(side*size*actual, stop_pct)
            logs.append({
                "DataDate":dates[i].strftime('%y%m%d'),
                "Date":dates[i].strftime('%Y-%m-%d'),
                "PredRet":pred,"Actual":actual,"Size":size,"Side":side,
                "Err":pred-actual,"SE":(pred-actual)**2,"PnL":pnl,
                "Hit":int(np.sign(pred)==np.sign(actual))
            })
        return pd.DataFrame(logs)
    best=None; best_sh=-np.inf
    total = len(cand_train) * len(cand_seq)
    for w in cand_train:
        for s in cand_seq:
            feats=_tech_indicators(high,low,price,vol,lb=max(14,s)).dropna()
            df=pd.concat([dates.rename('Date'),price.rename('Close'),ret.rename('Ret'),feats],axis=1).dropna()
            if len(df)<w+s+10:
                continue
            wf=_walk(df,w,s)
            if wf.empty:
                continue
            sh=wf['PnL'].mean()/wf['PnL'].std(ddof=0)*math.sqrt(252) if wf['PnL'].std(ddof=0)>0 else np.nan
            if sh>best_sh: best_sh, best=(sh,(w,s,wf,df))
    if best is None:
        return None, None
    WIN,SEQ,wf,df=best[1]
    rmse=math.sqrt(wf['SE'].mean())
    stats=dict(
        Sharpe=best_sh,
        BestWin=WIN,
        BestSeqLen=SEQ,
        RMSE=rmse
    )
    return wf, stats 