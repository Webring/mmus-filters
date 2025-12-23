import numpy as np


def mse(true, est):
    return np.mean((true - est) ** 2)


def rmse(true, est):
    return np.sqrt(mse(true, est))


def mae(true, est):
    return np.mean(np.abs(true - est))


def snr_db(true, est):
    noise = true - est
    return 10 * np.log10(np.var(true) / np.var(noise))


def metrics(true, noisy, filtered):
    mse_noisy = mse(true, noisy)
    mse_filt = mse(true, filtered)

    return {
        "MSE": mse_filt,
        "RMSE": rmse(true, filtered),
        "MAE": mae(true, filtered),
        "SNR": snr_db(true, filtered),
        "DELTA_MSE": (mse_noisy - mse_filt) / mse_noisy * 100,
    }
