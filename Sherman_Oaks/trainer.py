import tensorflow as tf
import numpy as np
import random
import pickle as pkl
from copy import deepcopy
import csv
from utils import merge_dicts
from tqdm import tqdm

class Trainer:
#### start here ####
    def __init__(self,
                 train_gen,
                 kl_weights,
                 n_epochs=1,
                 start_epoch=0,
                 val_gen=None,
                 monitor=None,
                 monitor_kl=None,
                 logger=None,
                ):
                
        self.train_gen = train_gen
        self.kl_weights = kl_weights
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch
        self.val_gen = val_gen
        self.monitor = monitor
        self.monitor_kl = monitor_kl
        self.logger = logger
        self.loss_dict = []

    def fit(self, model,):
                
        for epoch in range(self.start_epoch, self.n_epochs):
            # print('==== Epoch #{0:3d} ===='.format(epoch))
            model.kl_weight = self.kl_weights[epoch]

            with tqdm(self.train_gen, unit="batch") as tepoch:
                # for batch in tqdm(range(n_batches)):
                model.reset_metrics()  # reset metrics for training
                for x, y in tepoch:
                    # x, y = train_gen[batch]
                    tepoch.set_description(f"Epoch {epoch}")

                    y_r, y_p = y

                    with tf.GradientTape() as tape:  # Forward pass
                        recon_bottleneck = model.bottleneck_model(model.encoder_model(x, training=True))
                        pred_bottleneck = model.bottleneck_model(model.lstm_encoder(model.encoder_model(x, training=True)))

                        recon_sample = model.sampling_model(recon_bottleneck, training=True)
                        pred_sample = model.sampling_model(pred_bottleneck, training=True)

                        reconstruction = model.decoder_model(recon_sample, training=True)
                        prediction = model.decoder_model(pred_sample, training=True)

                        reconstruction_loss = tf.reduce_mean(
                            tf.math.square(y_r - reconstruction)
                        )

                        prediction_loss = tf.reduce_mean(
                            tf.math.square(y_p - prediction)
                        )

                        z_mean_r = recon_bottleneck[0]
                        z_log_var_r = recon_bottleneck[1]

                        z_mean_p = pred_bottleneck[0]
                        z_log_var_p = pred_bottleneck[1]

                        recon_kl_loss = -0.5 * (1 + z_log_var_r - tf.square(z_mean_r) - tf.exp(z_log_var_r))
                        recon_kl_loss = tf.reduce_mean(tf.reduce_sum(recon_kl_loss, axis=1))

                        pred_kl_loss = -0.5 * (1 + z_log_var_p - tf.square(z_mean_p) - tf.exp(z_log_var_p))
                        pred_kl_loss = tf.reduce_mean(tf.reduce_sum(pred_kl_loss, axis=1))

                        total_loss = reconstruction_loss + prediction_loss + model.kl_weight * recon_kl_loss + model.kl_weight * pred_kl_loss

                    grads = tape.gradient(total_loss, model.trainable_weights)
                    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    model.total_loss_tracker.update_state(total_loss)
                    model.reconstruction_loss_tracker.update_state(reconstruction_loss)
                    model.prediction_loss_tracker.update_state(prediction_loss)
                    model.recon_kl_loss_tracker.update_state(recon_kl_loss)
                    model.pred_kl_loss_tracker.update_state(pred_kl_loss)

                    losses = {
                        "loss": model.total_loss_tracker.result().numpy(),
                        "reconstruction_loss": model.reconstruction_loss_tracker.result().numpy(),
                        "prediction_loss": model.prediction_loss_tracker.result().numpy(),
                        "recon_kl_loss": model.recon_kl_loss_tracker.result().numpy(),
                        "pred_kl_loss": model.pred_kl_loss_tracker.result().numpy(),
                    }

                    tepoch.set_postfix(ordered_dict=losses)

                if self.val_gen is not None:

                    val_losses = self.evaluate(model)

                    merged_losses = merge_dicts(losses, val_losses)
                    
                else:
                    merged_losses = deepcopy(losses)
                       
                self.loss_dict.append(merged_losses)

                for key, value in merged_losses.items():
                    print(key + ":", str(value), end="\t")
                print('\n')

                if self.monitor is not None:
                    self.monitor.update(model, merged_losses)
                if self.monitor_kl is not None:
                    self.monitor_kl.update(model, merged_losses)

                self.logger.write_to_log(merged_losses)

                self.train_gen.on_epoch_end()
                
        return self.loss_dict
            
            
    def evaluate(self, model):
        model.reset_metrics()  # reset metrics for validation
        for x, y in self.val_gen:
            y_r, y_p = y

            recon_bottleneck = model.bottleneck_model(model.encoder_model(x, training=False))
            pred_bottleneck = model.bottleneck_model(model.lstm_encoder(model.encoder_model(x, training=False)))

            recon_sample = model.sampling_model(recon_bottleneck, training=False)
            pred_sample = model.sampling_model(pred_bottleneck, training=False)

            reconstruction = model.decoder_model(recon_sample, training=False)
            prediction = model.decoder_model(pred_sample, training=False)

            reconstruction_loss = tf.reduce_mean(
                tf.math.square(y_r - reconstruction)
            )

            prediction_loss = tf.reduce_mean(
                tf.math.square(y_p - prediction)
            )

            z_mean_r = recon_bottleneck[0]
            z_log_var_r = recon_bottleneck[1]

            z_mean_p = pred_bottleneck[0]
            z_log_var_p = pred_bottleneck[1]

            recon_kl_loss = -0.5 * (1 + z_log_var_r - tf.square(z_mean_r) - tf.exp(z_log_var_r))
            recon_kl_loss = tf.reduce_mean(tf.reduce_sum(recon_kl_loss, axis=1))

            pred_kl_loss = -0.5 * (1 + z_log_var_p - tf.square(z_mean_p) - tf.exp(z_log_var_p))
            pred_kl_loss = tf.reduce_mean(tf.reduce_sum(pred_kl_loss, axis=1))

            total_loss = reconstruction_loss + prediction_loss + model.kl_weight * recon_kl_loss + model.kl_weight * pred_kl_loss

            model.total_loss_tracker.update_state(total_loss)
            model.reconstruction_loss_tracker.update_state(reconstruction_loss)
            model.prediction_loss_tracker.update_state(prediction_loss)
            model.recon_kl_loss_tracker.update_state(recon_kl_loss)
            model.pred_kl_loss_tracker.update_state(pred_kl_loss)

        losses = {
            "val_loss": model.total_loss_tracker.result().numpy(),
            "val_reconstruction_loss": model.reconstruction_loss_tracker.result().numpy(),
            "val_prediction_loss": model.prediction_loss_tracker.result().numpy(),
            "val_recon_kl_loss": model.recon_kl_loss_tracker.result().numpy(),
            "val_pred_kl_loss": model.pred_kl_loss_tracker.result().numpy(),
        }

        return losses
