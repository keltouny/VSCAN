import numpy as np
from statistics import mean
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy


def RRSE(y_act, y_pred):
    top = np.sum(np.square((y_pred - y_act)), axis=-1)
    y_act_m = np.repeat(np.mean(y_act, axis=-1)[:, np.newaxis], repeats=y_act.shape[-1], axis=-1)
    bot = np.sum(np.square(y_act_m - y_act), axis=-1)

    loss = np.sqrt(top / bot)

    return loss


def evaluate_generator(test_gen, model, eval_func=RRSE, disc='test_mid_rrse'):
    di_tracker = []
    test_mid = []
    with tqdm(test_gen, unit="batch") as tepoch:
        for x, y in tepoch:
            tepoch.set_description(f"test_mid_rrse estimation")
            y_act, y_pred = model.bottleneck_output(x, y)
            di = eval_func(y_act, y_pred)
            test_mid.append(di)
            di_tracker.append(di.mean())
            tepoch.set_postfix(test_mid=mean(di_tracker))

    return test_mid

def plot_ui_final(type_name: str, dir_name: str, perc=99, fs=50):  # , std_factor=2.33) :
    val_mid = np.load(f"{dir_name}/val_{type_name}.npy")
    test_mid = np.load(f"{dir_name}/test_{type_name}.npy")

    val_mid_mean = np.mean(val_mid, axis=-1)  # , keepdims=True)
    test_mid_mean = np.mean(test_mid, axis=-1)  # , keepdims=True)

    val_mid_std = np.std(val_mid, axis=-1)  # , keepdims=True)
    test_mid_std = np.std(test_mid, axis=-1)  # , keepdims=True)

    val_mean = np.average(val_mid.reshape((-1,)))
    val_std = np.std(val_mid.reshape((-1,)))

    thold = np.percentile(val_mid.reshape((-1,)), perc)
    alpha = 1 - (perc / 100)

    val_detections = (val_mid_mean > thold).astype('int')
    detections = (test_mid_mean > thold).astype('int')

    val_ui = []

    for i in range(val_mid.shape[0]):
        if val_detections[i] == 1:
            val_ui.append(2 * norm(val_mid_mean[i], val_mid_std[i]).cdf(thold).item())
        else:
            val_ui.append(2 * (1 - norm(val_mid_mean[i], val_mid_std[i]).cdf(thold).item()))

    val_x = np.linspace(0, len(val_ui) / fs, len(val_ui))
    save_name = ''
    plt.figure(figsize=(15, 4), dpi=200)
    plt.plot(val_x, val_ui, 'b', label='val_ui', linewidth=0.5)
    # plt.title(f'Uncertainty Index')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Uncertainty Index')
    plt.ylim((0, 1))
    # plt.legend()
    plt.savefig(f'{dir_name}/val_{type_name}_ui_{save_name}_{perc}.png')
    plt.savefig(f'{dir_name}/val_{type_name}_ui_{save_name}_{perc}.svg')
    plt.close()

    ########################

    ui = []

    for i in range(test_mid.shape[0]):
        if detections[i] == 1:
            ui.append(2 * norm(test_mid_mean[i], test_mid_std[i]).cdf(thold).item())
        else:
            ui.append(2 * (1 - norm(test_mid_mean[i], test_mid_std[i]).cdf(thold).item()))

    test_x = np.linspace(0, len(ui) / fs, len(ui)).tolist()
    plt.figure(figsize=(15, 4), dpi=200)
    plt.plot(test_x, ui, 'b', label='ui', linewidth=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Uncertainty Index')
    plt.ylim((0, 1))
    # plt.legend()
    plt.savefig(f'{dir_name}/test_{type_name}_ui_{save_name}_{perc}.png')
    plt.savefig(f'{dir_name}/test_{type_name}_ui_{save_name}_{perc}.svg')
    plt.close()

    return (val_ui, ui)


def plot_di_stoch_final(type_name: str, dir_name: str, mode='samples', perc=99, std_factor=2.0, fs=50, log_scale=False):
    modes = ['samples', 'means', 'both']
    if mode not in modes:
        raise ValueError("Invalid sim type. Expected one of: %s" % modes)

    val_mid = np.load(f"{dir_name}/val_{type_name}.npy")
    test_mid = np.load(f"{dir_name}/test_{type_name}.npy")

    val_mid_mean = np.mean(val_mid, axis=-1)  # , keepdims=True)
    test_mid_mean = np.mean(test_mid, axis=-1)  # , keepdims=True)

    val_mid_std = np.std(val_mid, axis=-1)  # , keepdims=True)
    test_mid_std = np.std(test_mid, axis=-1)  # , keepdims=True)

    per99 = np.percentile(val_mid_mean, perc)
    per50 = np.percentile(val_mid_mean, 50)

    per99_total = np.percentile(val_mid.reshape((-1,)), 99)
    per50_total = np.percentile(val_mid.reshape((-1,)), 50)

    val_mean = np.mean(val_mid)
    val_std = np.std(val_mid)

    alpha = 1 - (perc / 100)
    z_critical = norm.ppf(1 - alpha)
    thold = val_mean + z_critical * val_std

    labels = [type_name]
    val_x = np.linspace(0, len(val_mid_mean) / fs, len(val_mid_mean)).tolist()
    val_std_up = val_mid_mean + std_factor * val_mid_std
    val_std_dn = val_mid_mean - std_factor * val_mid_std
    plt.figure(figsize=(15, 4), dpi=200)
    # for i in range(val_mid_mean.shape[1]):
    plt.fill_between(val_x, val_std_up, val_std_dn, color='lightgrey', label=f'Val. DI mean ±{std_factor}σ')
    plt.plot(val_x, val_mid_mean, 'b', linewidth=1, label='Val. DI mean')
    if log_scale:
        plt.yscale('log')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Damage Index')
    # plt.title(f'Validation {type_name} (Normal)')
    plt.ylim(top=np.max(test_mid_mean + std_factor * test_mid_std), bottom=0)
    plt.axhline(y=thold, color='r', linestyle='-.', label=f'Threshold at α={alpha:.2f}', linewidth=0.5)
    plt.axhline(y=val_mean, color='k', linestyle='-.', label='Reference DI mean', linewidth=0.5)
    plt.legend()

    if log_scale:
        plt.savefig(f"{dir_name}/val_{type_name}_std_log_final_{perc}.png")
    else:
        plt.savefig(f"{dir_name}/val_{type_name}_std_final_{perc}.png")
        plt.savefig(f"{dir_name}/val_{type_name}_std_final_{perc}.svg")
    plt.close()

    test_x = np.linspace(0, len(test_mid_mean) / fs, len(test_mid_mean)).tolist()

    test_std_up = test_mid_mean + std_factor * test_mid_std
    test_std_dn = test_mid_mean - std_factor * test_mid_std
    plt.figure(figsize=(15, 4), dpi=200)
    # for i in range(test_mid_mean.shape[1]):
    plt.fill_between(test_x, test_std_up, test_std_dn, color='lightgrey', label=f'DI mean ±{std_factor}σ')
    plt.plot(test_x, test_mid_mean, 'b', linewidth=1, label='DI mean')
    if log_scale:
        plt.yscale('log')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Damage Index')
    # plt.title(f'Validation {type_name} (Normal)')
    plt.ylim(top=np.max(test_mid_mean + std_factor * test_mid_std), bottom=0)
    plt.axhline(y=thold, color='r', linestyle='-.', label=f'Threshold at α={alpha:.2f}', linewidth=0.5)
    plt.axhline(y=val_mean, color='k', linestyle='-.', label='Reference DI mean', linewidth=0.5)
    plt.legend()

    if log_scale:
        plt.savefig(f"{dir_name}/test_{type_name}_std_log_final_{perc}.png")
    else:
        plt.savefig(f"{dir_name}/test_{type_name}_std_final_{perc}.png")
        plt.savefig(f"{dir_name}/test_{type_name}_std_final_{perc}.svg")
    plt.close()


def detections_rolling_stoch_final(type_name: str, dir_name: str, use_all_data=False, perc=99, fs=50,
                                   interval=1):  # , std_factor=2.33) :
    val_mid = np.load(f"{dir_name}/val_{type_name}.npy")
    test_mid = np.load(f"{dir_name}/test_{type_name}.npy")

    val_mid_mean = np.mean(val_mid, axis=-1)  # , keepdims=True)
    test_mid_mean = np.mean(test_mid, axis=-1)  # , keepdims=True)

    val_mean = np.mean(val_mid)
    val_std = np.std(val_mid)
    alpha = 1 - (perc / 100)
    z_critical = norm.ppf(1 - alpha)
    thold = val_mean + z_critical * val_std
    save_name = ''
    val_detections = (val_mid_mean > thold).astype('int')

    val_detections_rolling_1s = []
    for i in range(val_detections.shape[0] - interval * fs):
        val_detections_rolling_1s.append(np.sum(val_detections[i:i + interval * fs]))

    val_detections_rolling_1s = np.array(val_detections_rolling_1s)
    val_x = np.linspace(0, len(val_detections_rolling_1s) / fs, len(val_detections_rolling_1s)).tolist()

    plt.figure(figsize=(15, 4), dpi=200)
    plt.step(val_x, val_detections_rolling_1s, 'b', where='post')
    plt.title(
        f'validation detections in a rolling 1s; max = {val_detections_rolling_1s.max():.3f}; average = {val_detections_rolling_1s.mean():.3f}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Detections in a rolling 1-second')
    plt.ylim((0, fs))
    plt.savefig(f'{dir_name}/val_{type_name}_detections_rolling1s_{save_name}_final_{perc}.png')
    plt.savefig(f'{dir_name}/val_{type_name}_detections_rolling1s_{save_name}_final_{perc}.svg')
    plt.close()

    ########################

    detection_threshold_ratio = np.max(val_detections_rolling_1s)
    if detection_threshold_ratio == 0:
        detection_threshold_ratio = 1

    detections = (test_mid_mean > thold).astype('int')
    # detections_per_s = 50 * np.sum(detections)/detections.shape[0] #50hz

    detections_rolling_1s = []
    for i in range(detections.shape[0] - interval * fs):
        detections_rolling_1s.append(np.sum(detections[i:i + interval * fs]))

    detections_rolling_1s = np.array(detections_rolling_1s)
    test_x = np.linspace(0, len(detections_rolling_1s) / fs, len(detections_rolling_1s)).tolist()
    plt.figure(figsize=(15, 4), dpi=200)

    for i in range(detections_rolling_1s.shape[0]):

        if detections_rolling_1s[i] < detection_threshold_ratio:
            if i == detections_rolling_1s.shape[0] - 1:
                plt.plot([test_x[i], test_x[i], test_x[i] + (test_x[i - 1] - test_x[i])],
                         [detections_rolling_1s[i - 1], detections_rolling_1s[i], detections_rolling_1s[i]], 'b')
            elif i == 0:
                plt.plot([test_x[i], test_x[i], test_x[i + 1]], [0, detections_rolling_1s[i], detections_rolling_1s[i]],
                         'b')
            else:
                plt.plot([test_x[i], test_x[i], test_x[i + 1]],
                         [detections_rolling_1s[i - 1], detections_rolling_1s[i], detections_rolling_1s[i]], 'b')
        else:
            if i == detections_rolling_1s.shape[0] - 1:
                plt.plot([test_x[i], test_x[i], test_x[i] + (test_x[i - 1] - test_x[i])],
                         [detections_rolling_1s[i - 1], detections_rolling_1s[i], detections_rolling_1s[i]], 'r')
            elif i == 0:
                plt.plot([test_x[i], test_x[i], test_x[i + 1]], [0, detections_rolling_1s[i], detections_rolling_1s[i]],
                         'r')
            else:
                plt.plot([test_x[i], test_x[i], test_x[i + 1]],
                         [detections_rolling_1s[i - 1], detections_rolling_1s[i], detections_rolling_1s[i]], 'r')

    plt.title(
        f'test detections in a rolling 1s; max = {detections_rolling_1s.max():.3f}; average = {detections_rolling_1s.mean():.3f}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Detections in a rolling 1-second')
    plt.ylim((0, fs))
    plt.savefig(f'{dir_name}/test_{type_name}_detections_rolling1s_{save_name}_final_{perc}.png')
    plt.savefig(f'{dir_name}/test_{type_name}_detections_rolling1s_{save_name}_final_{perc}.svg')
    plt.close()

    return (val_detections_rolling_1s, detections_rolling_1s)

def detections_rolling_ui(ui, val_ui, type_name: str, dir_name: str, use_all_data=False, perc=99, ui_critical=0.4,
                              fs=50, interval=1, fillcolor='navajowhite'):  # , std_factor=2.33) :
    val_mid = np.load(f"{dir_name}/val_{type_name}.npy")
    test_mid = np.load(f"{dir_name}/test_{type_name}.npy")

    val_mid_mean = np.mean(val_mid, axis=-1)  # , keepdims=True)
    test_mid_mean = np.mean(test_mid, axis=-1)  # , keepdims=True)

    val_mean = np.mean(val_mid)
    val_std = np.std(val_mid)
    alpha = 1 - (perc / 100)
    z_critical = norm.ppf(1 - alpha)
    thold = val_mean + z_critical * val_std
    save_name = 'all'
    val_detections = (val_mid_mean > thold).astype('float')
    val_ui = np.array(val_ui)
    val_detections[np.where(val_ui > ui_critical)] = 0.5

    val_ui_detections = (val_ui > ui_critical).astype('int')

    val_detections_rolling_1s = []
    val_ui_detections_rolling_1s = []
    for i in range(val_detections.shape[0] - interval * fs):
        val_detections_rolling_1s.append(np.sum(val_detections[i:i + interval * fs]))
        val_ui_detections_rolling_1s.append(np.sum(val_ui_detections[i:i + interval * fs]))

    val_detections_rolling_1s = np.array(val_detections_rolling_1s)
    val_ui_detections_rolling_1s = np.array(val_ui_detections_rolling_1s)

    val_x = np.linspace(0, len(val_detections_rolling_1s) / fs, len(val_detections_rolling_1s)).tolist()

    plt.figure(figsize=(15, 4), dpi=200)
    # plt.step(val_x, val_detections_rolling_1s, 'b', where='post')

    for i in range(val_detections_rolling_1s.shape[0]):

        if i == val_detections_rolling_1s.shape[0] - 1:
            plt.fill_between([val_x[i], val_x[i] + (val_x[i - 1] - val_x[i])],
                             [val_ui_detections_rolling_1s[i], val_ui_detections_rolling_1s[i]], color=fillcolor)
        else:
            plt.fill_between([val_x[i], val_x[i + 1]],
                             [val_ui_detections_rolling_1s[i], val_ui_detections_rolling_1s[i]], color=fillcolor)

        if i == val_detections_rolling_1s.shape[0] - 1:
            plt.plot([val_x[i], val_x[i], val_x[i] + (val_x[i - 1] - val_x[i])],
                     [val_detections_rolling_1s[i - 1], val_detections_rolling_1s[i], val_detections_rolling_1s[i]],
                     'b')
        elif i == 0:
            plt.plot([val_x[i], val_x[i], val_x[i + 1]],
                     [0, val_detections_rolling_1s[i], val_detections_rolling_1s[i]], 'b')
        else:
            plt.plot([val_x[i], val_x[i], val_x[i + 1]],
                     [val_detections_rolling_1s[i - 1], val_detections_rolling_1s[i], val_detections_rolling_1s[i]],
                     'b')

    plt.title(
        f'validation detections in a rolling 1s; max = {val_detections_rolling_1s.max():.3f}; average = {val_detections_rolling_1s.mean():.3f}\nCritical UI commulative rolling 1s; max = {val_ui_detections_rolling_1s.max():.3f}; average = {val_ui_detections_rolling_1s.mean():.3f}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Detections in a rolling 1-second')
    plt.ylim((0, fs))
    plt.savefig(f'{dir_name}/val_{type_name}_detections_rolling1s_{save_name}_final_{perc}_fill.png')
    plt.savefig(f'{dir_name}/val_{type_name}_detections_rolling1s_{save_name}_final_{perc}_fill.svg')
    plt.close()

    ########################
    ui = np.array(ui)
    detection_threshold_ratio = np.max(val_detections_rolling_1s)
    if detection_threshold_ratio == 0:
        detection_threshold_ratio = 1

    detections = (test_mid_mean > thold).astype('float')
    ui_detections = (ui > ui_critical).astype('int')
    # make the corrections based on uncertainty
    detections[np.where(ui > ui_critical)] = 0.5
    # detections_per_s = 50 * np.sum(detections)/detections.shape[0] #50hz

    detections_rolling_1s = []
    ui_detections_rolling_1s = []
    for i in range(detections.shape[0] - interval * fs):
        detections_rolling_1s.append(np.sum(detections[i:i + interval * fs]))
        ui_detections_rolling_1s.append(np.sum(ui_detections[i:i + interval * fs]))

    detections_rolling_1s = np.array(detections_rolling_1s)
    ui_detections_rolling_1s = np.array(ui_detections_rolling_1s)
    test_x = np.linspace(0, len(detections_rolling_1s) / fs, len(detections_rolling_1s)).tolist()
    plt.figure(figsize=(15, 4), dpi=200)

    for i in range(detections_rolling_1s.shape[0]):

        if i == detections_rolling_1s.shape[0] - 1:
            plt.fill_between([test_x[i], test_x[i] + (test_x[i - 1] - test_x[i])],
                             [ui_detections_rolling_1s[i], ui_detections_rolling_1s[i]], color=fillcolor)
        else:
            plt.fill_between([test_x[i], test_x[i + 1]], [ui_detections_rolling_1s[i], ui_detections_rolling_1s[i]],
                             color=fillcolor)

        if detections_rolling_1s[i] < detection_threshold_ratio:
            if i == detections_rolling_1s.shape[0] - 1:
                plt.plot([test_x[i], test_x[i], test_x[i] + (test_x[i - 1] - test_x[i])],
                         [detections_rolling_1s[i - 1], detections_rolling_1s[i], detections_rolling_1s[i]], 'b')
            elif i == 0:
                plt.plot([test_x[i], test_x[i], test_x[i + 1]], [0, detections_rolling_1s[i], detections_rolling_1s[i]],
                         'b')
            else:
                plt.plot([test_x[i], test_x[i], test_x[i + 1]],
                         [detections_rolling_1s[i - 1], detections_rolling_1s[i], detections_rolling_1s[i]], 'b')
        else:
            if i == detections_rolling_1s.shape[0] - 1:
                plt.plot([test_x[i], test_x[i], test_x[i] + (test_x[i - 1] - test_x[i])],
                         [detections_rolling_1s[i - 1], detections_rolling_1s[i], detections_rolling_1s[i]], 'r')
            elif i == 0:
                plt.plot([test_x[i], test_x[i], test_x[i + 1]], [0, detections_rolling_1s[i], detections_rolling_1s[i]],
                         'r')
            else:
                plt.plot([test_x[i], test_x[i], test_x[i + 1]],
                         [detections_rolling_1s[i - 1], detections_rolling_1s[i], detections_rolling_1s[i]], 'r')

    plt.title(
        f'test detections in a rolling 1s; max = {detections_rolling_1s.max():.3f}; average = {detections_rolling_1s.mean():.3f}\nCritical UI commulative rolling 1s; max = {ui_detections_rolling_1s.max():.3f}; average = {ui_detections_rolling_1s.mean():.3f} ')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Detections in a rolling 1-second')
    plt.ylim((0, fs))
    plt.savefig(f'{dir_name}/test_{type_name}_detections_rolling1s_{save_name}_final_{perc}_UI_fill.png')
    plt.savefig(f'{dir_name}/test_{type_name}_detections_rolling1s_{save_name}_final_{perc}_UI_fill.svg')
    plt.close()

    return (val_detections_rolling_1s, detections_rolling_1s, ui_detections_rolling_1s)