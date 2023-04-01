import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Reshape, UpSampling2D, ConvLSTM2D, Activation, concatenate, ZeroPadding2D, Cropping2D, Cropping1D, LocallyConnected2D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softplus
from tensorflow import keras


def create_encoder(train_gen):
    batch_size = train_gen.batch_size
    in_time = train_gen.past_steps
    #out_time = train_gen.future_steps
    time_h2 = 4
    height = train_gen.hi
    width = train_gen.wi
    pad_h = 12 - height
    pad_w = 12 - width
    resp = train_gen.resp



    input_shape = Input(batch_shape=(batch_size, in_time, height, width, resp))
    input_shape_padding = TimeDistributed(ZeroPadding2D(padding=((pad_h, 0), (pad_w, 0))))(input_shape)
    encoder = TimeDistributed(Conv2D(filters=16,kernel_size=5, activation='tanh', padding='same', strides=(1,1)))(input_shape_padding)
    #encoder = TimeDistributed(BatchNormalization())(encoder)
    encoder = TimeDistributed(MaxPooling2D(pool_size=2))(encoder)
    encoder = TimeDistributed(Conv2D(filters=32,kernel_size=3, activation='tanh', padding='same', strides=(1,1)))(encoder)
    #encoder = TimeDistributed(BatchNormalization())(encoder)
    encoder = TimeDistributed(MaxPooling2D(pool_size=2))(encoder)
    encoder = TimeDistributed(Flatten())(encoder)
    encoder = TimeDistributed(Dense(64, activation='tanh',))(encoder)
    encoder_out = TimeDistributed(Dense(time_h2, activation='tanh',))(encoder)
    
    model = Model(input_shape, encoder_out, name='encoder_model')
 
    return model
 
def create_lstm_encoder(train_gen):

    batch_size = train_gen.batch_size
    in_time = train_gen.past_steps
    time_h2 = 4

    pred_input = Input(batch_shape=(batch_size, in_time, time_h2))
    prediction_input = LSTM(32, return_sequences=True)(pred_input)
    prediction_input = LSTM(64, return_sequences=True)(prediction_input)
    prediction_input = LSTM(128, return_sequences=True)(prediction_input)
    prediction_input = LSTM(128, return_sequences=True)(prediction_input)
    prediction_input = LSTM(64, return_sequences=True)(prediction_input)
    pred_out = LSTM(time_h2, return_sequences=True)(prediction_input)
    
    model = Model(pred_input, pred_out, name='lstm_encoder')

    return model    
    
    
def create_bottleneck_model(train_gen):

    batch_size = train_gen.batch_size
    in_time = train_gen.past_steps
    out_time = train_gen.future_steps
    time_h2 = 4
   
    
    encoder_out = Input(batch_shape=(batch_size, in_time, time_h2))

    bottleneck = Reshape((in_time * time_h2,))(encoder_out)
    bottleneck_mu = Dense(out_time * time_h2, activation='linear')(bottleneck) # change from 20 to 100
    bottleneck_log_variance = Dense(out_time * time_h2, activation=softplus)(bottleneck)
    
    model = Model(encoder_out, [bottleneck_mu, bottleneck_log_variance], name='bottleneck_model')
    
    return model


class sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
def create_sampling_model(train_gen):

    batch_size = train_gen.batch_size
    out_time = train_gen.future_steps
    time_h2 = 4

    mu_input = Input(batch_shape=(batch_size, out_time * time_h2))
    log_variance_input = Input(batch_shape=(batch_size, out_time * time_h2))
    
    latent_sample = sampling()([mu_input, log_variance_input])
    
    # latent_sample = Lambda(sampling, name="latent_sample")([mu_input, log_variance_input])
    
    model = Model([mu_input, log_variance_input], latent_sample, name="sampling_model")
    
    return model
   
    
def create_decoder(train_gen):

    batch_size = train_gen.batch_size
    out_time = train_gen.future_steps
    time_h2 = 4
    height = train_gen.hi
    width = train_gen.wi
    pad_h = 12 - height
    pad_w = 12 - width
    resp = train_gen.resp

    decoder_input = Input(batch_shape=(batch_size, out_time * time_h2)) # change from 20 to 100
    decoder = Reshape((out_time, time_h2))(decoder_input)

    decoder = TimeDistributed(Dense(64, activation='tanh',))(decoder)
    decoder = TimeDistributed(Dense(288, activation='tanh',))(decoder)
    decoder = TimeDistributed(Reshape((3,3,32)))(decoder)
    decoder = TimeDistributed(UpSampling2D(2))(decoder)
    decoder = TimeDistributed(Conv2D(filters=32,kernel_size=3, activation='tanh', padding='same'))(decoder) #19x-x32x32x16
    #decoder = TimeDistributed(BatchNormalization())(decoder)
    decoder = TimeDistributed(UpSampling2D(2))(decoder)
    decoder = TimeDistributed(Conv2D(filters=16,kernel_size=5, activation='tanh', padding='same'))(decoder) #19x-x32x32x16
    #decoder = TimeDistributed(BatchNormalization())(decoder)
    decoder = TimeDistributed(Cropping2D(cropping=((pad_h, 0), (pad_w, 0))))(decoder)
    decoder = TimeDistributed(Conv2D(filters=resp,kernel_size=1, activation='linear', padding='same'))(decoder)
    decoder_out = TimeDistributed(LocallyConnected2D(filters=resp,kernel_size=(1,1), activation='linear', padding='valid'))(decoder)

    model = Model(decoder_input, decoder_out, name='decoder_model')
    
    return model


class VAE(Model):
    def __init__(self, encoder_model, lstm_encoder, bottleneck_model, sampling_model, decoder_model, kl_weight=0.02,**kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder_model = encoder_model
        self.lstm_encoder = lstm_encoder
        self.bottleneck_model = bottleneck_model
        self.sampling_model = sampling_model
        self.decoder_model = decoder_model
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.prediction_loss_tracker = keras.metrics.Mean(
            name="prediction_loss"
        )
        self.recon_kl_loss_tracker = keras.metrics.Mean(name="recon_kl_loss")
        self.pred_kl_loss_tracker = keras.metrics.Mean(name="pred_kl_loss")
        
        
        self.kl_weight = kl_weight
        


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.prediction_loss_tracker,
            self.recon_kl_loss_tracker,
            self.pred_kl_loss_tracker,
        ]
        
    def call(self, input):
        x = input
        
        recon_bottleneck = self.bottleneck_model(self.encoder_model(x))
        pred_bottleneck = self.bottleneck_model(self.lstm_encoder(self.encoder_model(x)))
        
        recon_sample = self.sampling_model(recon_bottleneck)
        pred_sample = self.sampling_model(pred_bottleneck)
        
        
        reconstruction = self.decoder_model(recon_sample)
        prediction = self.decoder_model(pred_sample)
        
        return [reconstruction, prediction]


    @tf.function
    def train_step(self, data):
    
        x, y = data
        y_r, y_p = y
        # mse = tf.keras.losses.MeanSquaredError()
    
        with tf.GradientTape() as tape:
            
            recon_bottleneck = self.bottleneck_model(self.encoder_model(x))
            pred_bottleneck = self.bottleneck_model(self.lstm_encoder(self.encoder_model(x)))
            
            recon_sample = self.sampling_model(recon_bottleneck)
            pred_sample = self.sampling_model(pred_bottleneck)
            
            
            reconstruction = self.decoder_model(recon_sample)
            prediction = self.decoder_model(pred_sample)
           
            
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
            recon_kl_loss = self.kl_weight*tf.reduce_mean(tf.reduce_sum(recon_kl_loss, axis=1))
            
            pred_kl_loss = -0.5 * (1 + z_log_var_p - tf.square(z_mean_p) - tf.exp(z_log_var_p))
            pred_kl_loss = self.kl_weight*tf.reduce_mean(tf.reduce_sum(pred_kl_loss, axis=1))            
            
            
            total_loss = reconstruction_loss + prediction_loss + recon_kl_loss + pred_kl_loss
            
            
            
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.recon_kl_loss_tracker.update_state(recon_kl_loss)
        self.pred_kl_loss_tracker.update_state(pred_kl_loss)
        
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "recon_kl_loss": self.recon_kl_loss_tracker.result(),
            "pred_kl_loss": self.pred_kl_loss_tracker.result(),
        }
        
    @tf.function    
    def test_step(self, data):
    
        x, y = data
        y_r, y_p = y
               
        recon_bottleneck = self.bottleneck_model(self.encoder_model(x, training=False), training=False)
        pred_bottleneck = self.bottleneck_model(self.lstm_encoder(self.encoder_model(x, training=False), training=False), training=False)
        
        recon_sample = self.sampling_model(recon_bottleneck, training=False)
        pred_sample = self.sampling_model(pred_bottleneck, training=False)
        
        
        reconstruction = self.decoder_model(recon_sample, training=False)
        prediction = self.decoder_model(pred_sample, training=False)
        
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
        recon_kl_loss = self.kl_weight*tf.reduce_mean(tf.reduce_sum(recon_kl_loss, axis=1))
        
        pred_kl_loss = -0.5 * (1 + z_log_var_p - tf.square(z_mean_p) - tf.exp(z_log_var_p))
        pred_kl_loss = self.kl_weight*tf.reduce_mean(tf.reduce_sum(pred_kl_loss, axis=1))            
        
        
        total_loss = reconstruction_loss + prediction_loss + recon_kl_loss + pred_kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.recon_kl_loss_tracker.update_state(recon_kl_loss)
        self.pred_kl_loss_tracker.update_state(pred_kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "recon_kl_loss": self.recon_kl_loss_tracker.result(),
            "pred_kl_loss": self.pred_kl_loss_tracker.result(),
        }
        
    def bottleneck_output(self, x, y):

        recon_bottleneck = self.bottleneck_model( self.encoder_model(y, training=False))
        pred_bottleneck = self.bottleneck_model(self.lstm_encoder( self.encoder_model(x, training=False)))

        recon_sample = self.sampling_model(recon_bottleneck, training=False).numpy()
        pred_sample = self.sampling_model(pred_bottleneck, training=False).numpy()

        return (recon_sample, pred_sample)
        

def create_modelSCAN(train_gen, kl_weight=0.02):
    batch_size = train_gen.batch_size
    in_time = train_gen.past_steps
    out_time = train_gen.future_steps
    time_h2 = 4
    height = train_gen.hi
    width = train_gen.wi
    pad_h = 12 - height
    pad_w = 12 - width
    resp = train_gen.resp

    input_shape = (batch_size, in_time, height, width, resp)

    encoder_model = create_encoder(train_gen)
    bottleneck_model = create_bottleneck_model(train_gen) 
    sampling_model = create_sampling_model(train_gen)
    lstm_encoder = create_lstm_encoder(train_gen) 
    decoder_model = create_decoder(train_gen)
    
    model = VAE(encoder_model, lstm_encoder, bottleneck_model, sampling_model, decoder_model, kl_weight)
    
    
    #
    model.compile(optimizer=Adam(learning_rate=5e-4, epsilon=1e-7))
    #
    model.build(input_shape)
    #
    
    model.summary()

    return model
    
