import os
import numpy as np
import matplotlib.pyplot as plt

from DataGenerators_SCAN import TrainGenerator, EvalGenerator, EvalGeneratorMidStoch
from model_vSCAN_trainer import create_modelSCAN
from trainer import *
from utils import *
from detection import *
import tensorflow as tf
import shutil

tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

tf.keras.backend.clear_session()

# from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


trial = 101
trial_desc = 'Van Nuys - vSCAN'

train_model = False
test_model = True
verbose = 1

print(f'TRIAL {trial}\n')
print(f'trial_desc: {trial_desc}')


work_directory = os.getcwd()
model_dir = 'F:/model_saves/P6'

dir_name = f'{work_directory}/output_vSCAN{trial}'
model_dir = f'{model_dir}/output{trial}'
trial_type = 'csmip'

if not os.path.exists(work_directory):
    print('WARNING! WORK DIRECTORY DOES NOT EXIST... CREATING A NEW ONE')
    os.mkdir(work_directory)
os.chdir(work_directory)

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

with open(f'{dir_name}/trial_desc.txt', "w") as f:
    f.write(trial_desc)


epoch_size = 256
past_steps = 75

height = 9
width = 10
response = 2
all_channel_address = [[0, 2, 1], [8, 2, 1], [8, 9, 1], [6, 9, 1], [3, 2, 1], [3, 9, 1], [2, 2, 1], [2, 9, 1],
                       [8, 9, 0],
                       [6, 9, 0], [3, 9, 0], [2, 9, 0], [0, 9, 0]]
test_ids = [2]
val_ids = [0]
train_avoid_ids = [0,2]
train_gen = TrainGenerator(val_ids=train_avoid_ids,
                           epoch_size=epoch_size,
                           past_steps=past_steps,
                           future_steps=100-past_steps,
                           normalize=False,
                           height=height,
                           width=width,
                           response=response,
                           all_channel_address=all_channel_address)

val_gen = EvalGenerator(batch_size=32,
                        val_ids=val_ids,
                        epoch_size=epoch_size,
                        past_steps=past_steps,
                        future_steps=100 - past_steps,
                        normalize=False,
                        height=height,
                        width=width,
                        response=response,
                        all_channel_address=all_channel_address)

test_gen = EvalGenerator(batch_size=32,
                         val_ids=test_ids,
                         epoch_size=epoch_size,
                         past_steps=past_steps,
                         future_steps=100 - past_steps,
                         normalize=False,
                         height=height,
                         width=width,
                         response=response,
                         all_channel_address=all_channel_address)

# model.load_weights(f'{dir_name}/saved-weights_csmip__606_first_trial.hdf5')


n_epochs = 500
# loss_dict = []
kl_start = 0
kl_weight_final = 0.05
warmup_epochs = 100
kl_weights = create_kl_weight_list(kl_weight_final, warmup_epochs, total_epochs=n_epochs, mode='sigmoid')

model = create_modelSCAN(train_gen, kl_weight=kl_weight_final)
model.kl_final = kl_weight_final

weightspath = f"{model_dir}/saved-weights_{trial_type}_{trial}.hdf5"
weightspath_kl = f"{model_dir}/saved-weights_weightedKL_{trial_type}_{trial}.hdf5"

# start_epoch = 0

# model.load_weights('output_GAT610/saved-weights_csmip__610.hdf5')


csv_logger_file = f'{dir_name}/log_{trial}.csv'

logger = CustomCSVLogger(csv_logger_file)
monitor_list = ['val_reconstruction_loss', 'val_prediction_loss',  'val_recon_kl_loss', 'val_pred_kl_loss']
monitor_list_kl = ['val_loss']

monitor = ModelMonitor(weightspath, monitor_list, mode='min', disc='acc. to unweighted KL')
monitor_kl = ModelMonitor(weightspath_kl, monitor_list_kl, mode='min', disc='acc. to weighted KL')

trainer = Trainer(train_gen,
                 kl_weights=kl_weights,
                 n_epochs=n_epochs,
                 val_gen=val_gen,
                 monitor=monitor,
                 monitor_kl=monitor_kl,
                 logger=logger,)

if train_model:
    loss_dict = trainer.fit(model)


    weightspath_old = f"{model_dir}/saved-weights_{trial_type}_{trial}.hdf5"
    weightspath = f"{dir_name}/saved-weights_{trial_type}_{trial}.hdf5"
    shutil.copyfile(weightspath_old, weightspath)

    weightspath_kl_old = f"{model_dir}/saved-weights_weightedKL_{trial_type}_{trial}.hdf5"
    weightspath_kl = f"{dir_name}/saved-weights_weightedKL_{trial_type}_{trial}.hdf5"
    shutil.copyfile(weightspath_kl_old, weightspath_kl)

    model.save_weights(f"{dir_name}/saved-weights_{trial_type}_{trial}_final.hdf5")

    loss_list = {key: [i[key] for i in loss_dict] for key in loss_dict[0]}

    # Plotting losses and accuracies at each epoch
    plt.figure()
    #    plt.subplots_adjust(hspace=1.0)
    #    plt.subplot(2, 1, 1)
    # plt.axis([0, len(history.history['loss']), 0, 1])
    plt.plot(loss_list['loss'], label='Trainig set')
    plt.plot(loss_list['val_loss'], label='Validation set')
    plt.title('total loss vs epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.grid(b=True, which='major', color='darkgrey', linestyle='-')
    plt.grid(b=True, which='minor', color='grey', linestyle='-', alpha=0.2)
    plt.minorticks_on()

    plt.savefig(f'{dir_name}/graph_{trial_type}{trial}')

    # Plotting losses and accuracies at each epoch
    plt.figure()
    #    plt.subplots_adjust(hspace=1.0)
    #    plt.subplot(2, 1, 1)
    # plt.axis([0, len(history.history['loss']), 0, 1])
    plt.plot(loss_list['recon_kl_loss'], label='recon_kl_loss')
    plt.plot(loss_list['pred_kl_loss'], label='pred_kl_loss')
    plt.plot(loss_list['val_recon_kl_loss'], label='val_recon_kl_loss')
    plt.plot(loss_list['val_pred_kl_loss'], label='val_pred_kl_loss')
    plt.title('kl loss vs epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.grid(b=True, which='major', color='darkgrey', linestyle='-')
    plt.grid(b=True, which='minor', color='grey', linestyle='-', alpha=0.2)
    plt.minorticks_on()

    plt.savefig(f'{dir_name}/graph_kl_loss_{trial_type}{trial}')

    # Plotting losses and accuracies at each epoch
    plt.figure()
    #    plt.subplots_adjust(hspace=1.0)
    #    plt.subplot(2, 1, 1)
    # plt.axis([0, len(history.history['loss']), 0, 1])
    plt.plot(loss_list['reconstruction_loss'], label='reconstruction_loss')
    plt.plot(loss_list['prediction_loss'], label='prediction_loss')
    plt.plot(loss_list['val_reconstruction_loss'], label='val_reconstruction_loss')
    plt.plot(loss_list['val_prediction_loss'], label='val_prediction_loss')
    plt.title('mse loss vs epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.grid(b=True, which='major', color='darkgrey', linestyle='-')
    plt.grid(b=True, which='minor', color='grey', linestyle='-', alpha=0.2)
    plt.minorticks_on()

    plt.savefig(f'{dir_name}/graph_mse_loss_{trial_type}{trial}')

    save_to_csv(f'{dir_name}/log_{trial}_final.csv', loss_dict)

    save_pkl(f'{dir_name}/losses_{trial}', loss_dict)

    model.save_weights(f"{dir_name}/saved-weights_{trial_type}_{trial}_final.hdf5")


if test_model:
    ##### LOAD MODEL ####

    weightspath = f"{dir_name}/saved-weights_{trial_type}_{trial}_final.hdf5"
    model.load_weights(weightspath)


    ####################################################################
    ### stochastic output ####

    type_name = 'mid_rrse_32'
    batch_size = 32

    val_gen = EvalGeneratorMidStoch(batch_size=batch_size,
                                    val_ids=val_ids,
                                    epoch_size=epoch_size,
                                    past_steps=past_steps,
                                    future_steps=100 - past_steps,
                                    normalize=False,
                                    height=height,
                                    width=width,
                                    response=response,
                                    all_channel_address=all_channel_address)

    test_gen = EvalGeneratorMidStoch(batch_size=batch_size,
                                     val_ids=test_ids,
                                     epoch_size=epoch_size,
                                     past_steps=past_steps,
                                     future_steps=100 - past_steps,
                                     normalize=False,
                                     height=height,
                                     width=width,
                                     response=response,
                                     all_channel_address=all_channel_address)

    val_mid = evaluate_generator(val_gen, model, disc=f'val_{type_name}')
    test_mid = evaluate_generator(test_gen, model, disc=f'test_{type_name}')


    val_mid = np.array(val_mid)
    test_mid = np.array(test_mid)

    val_mid_rrse_mean = np.mean(val_mid, axis=-1, keepdims=True)
    test_mid_rrse_mean = np.mean(test_mid, axis=-1, keepdims=True)

    val_mid_rrse_std = np.std(val_mid, axis=-1, keepdims=True)
    test_mid_rrse_std = np.std(test_mid, axis=-1, keepdims=True)

    np.save(f"{dir_name}/val_{type_name}.npy", val_mid)
    np.save(f"{dir_name}/test_{type_name}.npy", test_mid)

    np.save(f"{dir_name}/val_{type_name}_mean.npy", val_mid_rrse_mean)
    np.save(f"{dir_name}/test_{type_name}_mean.npy", test_mid_rrse_mean)

    np.save(f"{dir_name}/val_{type_name}_std.npy", val_mid_rrse_std)
    np.save(f"{dir_name}/test_{type_name}_std.npy", test_mid_rrse_std)

    ###########################

    type_name = 'mid_rrse_32'

    plot_di_stoch_final(type_name, dir_name=dir_name, perc=95,)
    val_ui, ui = plot_ui_final(type_name, dir_name=dir_name, perc=95,)
    val_detections_rolling_1s, detections_rolling_1s = detections_rolling_stoch_final(type_name, dir_name=dir_name, perc=95,)
    val_detections_rolling_1s, detections_rolling_1s, ui_detections_rolling_1s = detections_rolling_ui(ui, val_ui, type_name, dir_name=dir_name, perc=95,)

