import numpy as np
import pandas as pd

data_path = './data/kr_close_.csv'
data_df = pd.read_csv(data_path, index_col=0)


def log_y_nd(log_p, n):
    assert len(log_p.shape) == 2
    r, c = log_p.shape

    return np.r_[log_p[:n, :] - log_p[:1, :], log_p[n:, :] - log_p[:-n, :]]

def fft(log_p, n):
    log_p_fft = np.fft.fft(log_p, axis=0)
    log_p_fft[n:-n] = 0
    return np.real(np.fft.ifft(log_p_fft, axis=0))

def std_nd(log_p, n):
    y = np.exp(log_y_nd(log_p, 1)) - 1.
    stdarr = np.zeros_like(y)
    for t in range(1, len(y)):
        stdarr[t, :] = np.std(y[max(0, t - n):(t + 1), :], axis=0)

    return stdarr

def processing(df):
    if len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(0)

    df_selected = df[(df.index > '1990-01-01') & (df.index <= '1991-01-01')]
    df_not_null = df_selected.ix[:, np.sum(df_selected.isna(), axis=0) == 0]
    infocodes = list(df_not_null.columns)
    dates = list(df_not_null.index)

    n_dates, n_infocode = df_not_null.shape

    log_p = np.log(df_not_null.values, dtype=np.float32)
    log_1y = log_y_nd(log_p, 1)
    log_5y = log_y_nd(log_p, 5)
    log_20y = log_y_nd(log_p, 20)
    log_60y = log_y_nd(log_p, 60)
    log_120y = log_y_nd(log_p, 120)
    log_240y = log_y_nd(log_p, 240)

    fft_3com = fft(log_p, 3)
    fft_6com = fft(log_p, 6)
    fft_100com = fft(log_p, 100)

    std_20 = std_nd(log_p, 20)




