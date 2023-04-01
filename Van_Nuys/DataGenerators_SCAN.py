import tensorflow as tf
import numpy as np
import random
import pickle as pkl
from copy import deepcopy

from tensorflow.keras.utils import Sequence

class TrainGenerator(Sequence):

    def __init__(self,
                 data_path='all_records_decimated',
                 # adj_path='all_adj_adjusted',
                 batch_size=32,
                 epoch_size=256,
                 past_steps=50,
                 future_steps=50,
                 n_models=1,
                 validate=False,
                 val_ids=[0,2],
                 normalize=False,
                 height=9,
                 width=10,
                 response=2,
                 all_channel_address=[[0, 2, 1], [8, 2, 1], [8, 9, 1], [6, 9, 1], [3, 2, 1], [3, 9, 1], [2, 2, 1], [2, 9, 1], [8, 9, 0],
                                      [6, 9, 0], [3, 9, 0], [2, 9, 0], [0, 9, 0]],
                 seed=1000, ):

        'Initialization'

        with open(f'{data_path}.pkl', 'rb') as handle:
            self.data_raw = pkl.load(handle)


        self.data = []

        for event in self.data_raw:
            self.data.append(event['record']['records'])
        self.maxch = len(all_channel_address)
        for idx in range(len(self.data)):
            del self.data[idx][self.maxch:]

        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.n_models = n_models
        self.seed = seed
        self.validate = validate
        self.val_ids = val_ids
        self.on_epoch_end()
        self.n_act = len(self.data)
        self.normalize = normalize
        self.all_channel_address = all_channel_address
        self.hi = height
        self.wi = width
        self.resp = response

        # self.n_ch = self.data[0].shape[1]

        random.seed(self.seed)

        self.seq_len = self.past_steps + self.future_steps

        self.channels = []
        self.points = []
        n_events = len(self.data)
        # self.n_val = 2
        # self.n_train = n_events-self.n_val
        # self.val_id = [0]
        # self.test_id = [2]
        # self.train_id = [1, 3]
        self.train_ids = []

        for idx in range(len(self.data)):
            if idx not in self.val_ids:
                self.train_ids.append(idx)



        if self.validate:
            self.data = [self.data[idx] for idx in self.val_ids]
            self.epoch_size = len(self.val_ids)

        else:
            self.data = [self.data[idx] for idx in self.train_ids]

        self.missing_sensors = []
        for k in range(self.resp):
            miss_sens_resp = []
            counter = 0
            for i in range(self.hi):
                for j in range(self.wi):
                    miss_address = [i,j,k]
                    if miss_address not in self.all_channel_address:
                        miss_sens_resp.append(deepcopy(counter))
                    counter+=1
            self.missing_sensors.append(deepcopy(miss_sens_resp))

        self.events = [x for x in range(len(self.data))]
        self.data_arr = []
        for idx, event in enumerate(self.data):
            self.channels.append(len(event))
            all_len = []
            for ch in range(len(event)):
                all_len.append(len(event[ch]))
            min_len = min(all_len)
            self.points.append(deepcopy(min_len))
            for ch in range(len(event)):
                del self.data[idx][ch][min_len:]

            event_grid = np.zeros((min_len, self.hi, self.wi, self.resp))
            event_arr = np.array(event).swapaxes(0, 1)

            for ch in range(self.maxch):
                ch_add = self.all_channel_address[ch]
                event_grid[:,ch_add[0],ch_add[1],ch_add[2]] = event_arr[:,ch]
            self.data_arr.append(event_grid)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.epoch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        event_id = random.choices(self.events, weights=self.points, k=1)[0]
        # Generate data
        data_tuple = self.__data_generation(event_id)
        return data_tuple

    def on_epoch_end(self):  # Edited by KE
        'Updates indexes after each epoch'
        pass

    def __data_generation(self, event_id):  # Edited by KE
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        n_ch = self.channels[event_id]
        seq = np.zeros((self.batch_size * self.n_models, self.seq_len, self.hi, self.wi, self.resp))
        # Generate data

        full_seq = self.data_arr[event_id]

        for i in range(int(self.batch_size * self.n_models)):
            start = random.randrange(full_seq.shape[0] - self.seq_len)
            seq[i] = full_seq[start:start + self.seq_len]

        seq = seq.reshape((self.batch_size, self.n_models, self.seq_len, self.hi, self.wi, self.resp))

        seq = np.moveaxis(seq, 1, -2).reshape((self.batch_size, self.seq_len, self.hi, self.wi, -1))

        if self.normalize:
            rms = np.average(np.square(seq))
            seq = seq/np.sqrt(rms)

        # mu = 0
        # sigma = 1E-10  # cm
        # noise = np.random.normal(mu, sigma, self.batch_size * self.past_steps * n_ch * self.n_models).reshape(
        #     (self.batch_size, -1, self.past_steps,))

        y_r = seq[:,self.past_steps-self.future_steps:self.past_steps]
        y_p = seq[:, self.past_steps:]

        x = seq[:, :self.past_steps]

        y = [y_r, y_p]

        return (x, y)


class EvalGenerator(Sequence):

    def __init__(self,
                 data_path='all_records_decimated',
                 # adj_path='all_adj_adjusted',
                 batch_size=32,
                 epoch_size=256,
                 past_steps=50,
                 future_steps=50,
                 n_models=1,
                 validate=True,
                 val_ids=[0],
                 normalize=True,
                 height=9,
                 width=10,
                 response=2,
                 all_channel_address=[[0, 2, 1], [8, 2, 1], [8, 9, 1], [6, 9, 1], [3, 2, 1], [3, 9, 1], [2, 2, 1],
                                      [2, 9, 1], [8, 9, 0],
                                      [6, 9, 0], [3, 9, 0], [2, 9, 0], [0, 9, 0]],
                 seed=1000, ):

        'Initialization'

        with open(f'{data_path}.pkl', 'rb') as handle:
            self.data_raw = pkl.load(handle)

        self.data = []

        for event in self.data_raw:
            self.data.append(event['record']['records'])
        self.maxch = len(all_channel_address)
        for idx in range(len(self.data)):
            del self.data[idx][self.maxch:]

        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.n_models = n_models
        self.seed = seed
        self.validate = validate
        self.val_ids = val_ids
        self.on_epoch_end()
        self.n_act = len(self.data)
        self.normalize = normalize
        self.all_channel_address = all_channel_address
        self.hi = height
        self.wi = width
        self.resp = response

        random.seed(self.seed)

        self.seq_len = self.past_steps + self.future_steps

        self.channels = []
        self.points = []
        n_events = len(self.data)
        self.train_ids = []

        for idx in range(len(self.data)):
            if idx not in self.val_ids:
                self.train_ids.append(idx)

        if self.validate:
            self.data = [self.data[idx] for idx in self.val_ids]
            self.epoch_size = len(self.val_ids)

        else:
            self.data = [self.data[idx] for idx in self.train_ids]

        self.missing_sensors = []
        for k in range(self.resp):
            miss_sens_resp = []
            counter = 0
            for i in range(self.hi):
                for j in range(self.wi):
                    miss_address = [i, j, k]
                    if miss_address not in self.all_channel_address:
                        miss_sens_resp.append(deepcopy(counter))
                    counter += 1
            self.missing_sensors.append(deepcopy(miss_sens_resp))

        self.events = [x for x in range(len(self.data))]
        self.data_arr = []

        self.batch_lengths = []
        self.total_batch_length = 0

        for idx, event in enumerate(self.data):
            self.channels.append(len(event))
            all_len = []
            for ch in range(len(event)):
                all_len.append(len(event[ch]))
            min_len = min(all_len)
            self.points.append(deepcopy(min_len))

            batch_length = (min_len - self.seq_len) // self.batch_size

            self.total_batch_length += batch_length
            self.batch_lengths.append(deepcopy(self.total_batch_length))

            for ch in range(len(event)):
                del self.data[idx][ch][min_len:]

            event_grid = np.zeros((min_len, self.hi, self.wi, self.resp))
            event_arr = np.array(event).swapaxes(0, 1)

            for ch in range(self.maxch):
                ch_add = self.all_channel_address[ch]
                event_grid[:, ch_add[0], ch_add[1], ch_add[2]] = event_arr[:, ch]
            self.data_arr.append(event_grid)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.total_batch_length

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        access_id = sum(i - 1 < index for i in self.batch_lengths)

        event_id = self.events[access_id]

        if access_id > 0:
            new_index = index - self.batch_lengths[access_id-1]
        else:
            new_index = index

        # print(event_id)
        # Generate data
        data_tuple = self.__data_generation(new_index, event_id)
        return data_tuple

    def on_epoch_end(self):  # Edited by KE
        'Updates indexes after each epoch'
        pass

    def __data_generation(self, index, event_id):  # Edited by KE
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        n_ch = self.channels[event_id]
        seq = np.zeros((self.batch_size, self.n_models, self.seq_len, self.hi, self.wi, self.resp))
        # Generate data
        for i in range(int(self.batch_size)):
            start = index*self.batch_size + i
            seq_single = self.data_arr[event_id][start:start + self.seq_len]
            seq[i] = np.repeat(seq_single[np.newaxis], repeats=self.n_models, axis=0)

        seq = np.moveaxis(seq, 1, -2).reshape((self.batch_size, self.seq_len, self.hi, self.wi, -1))

        if self.normalize:
            rms = np.average(np.square(seq))
            seq = seq / np.sqrt(rms)

        # mu = 0
        # sigma = 1E-10  # cm
        # noise = np.random.normal(mu, sigma, self.batch_size * self.past_steps * n_ch * self.n_models).reshape(
        #     (self.batch_size, -1, self.past_steps,))

        y_r = seq[:, self.past_steps - self.future_steps:self.past_steps]
        y_p = seq[:, self.past_steps:]

        x = seq[:, :self.past_steps]

        y = [y_r, y_p]

        return (x, y)


class EvalGeneratorMidStoch(Sequence):

    def __init__(self,
                 data_path='all_records_decimated',
                 # adj_path='all_adj_adjusted',
                 batch_size=32,
                 epoch_size=256,
                 past_steps=50,
                 future_steps=50,
                 n_models=1,
                 validate=True,
                 val_ids=[0],
                 normalize=True,
                 height=9,
                 width=10,
                 response=2,
                 all_channel_address=[[0, 2, 1], [8, 2, 1], [8, 9, 1], [6, 9, 1], [3, 2, 1], [3, 9, 1], [2, 2, 1],
                                      [2, 9, 1], [8, 9, 0],
                                      [6, 9, 0], [3, 9, 0], [2, 9, 0], [0, 9, 0]],
                 seed=1000, ):

        'Initialization'

        with open(f'{data_path}.pkl', 'rb') as handle:
            self.data_raw = pkl.load(handle)

        self.data = []

        for event in self.data_raw:
            self.data.append(event['record']['records'])
        self.maxch = len(all_channel_address)
        for idx in range(len(self.data)):
            del self.data[idx][self.maxch:]

        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.n_models = n_models
        self.seed = seed
        self.validate = validate
        self.val_ids = val_ids
        self.on_epoch_end()
        self.n_act = len(self.data)
        self.normalize = normalize
        self.all_channel_address = all_channel_address
        self.hi = height
        self.wi = width
        self.resp = response

        random.seed(self.seed)

        self.seq_len = self.past_steps + self.future_steps

        self.channels = []
        self.points = []
        n_events = len(self.data)
        self.train_ids = []

        for idx in range(len(self.data)):
            if idx not in self.val_ids:
                self.train_ids.append(idx)

        if self.validate:
            self.data = [self.data[idx] for idx in self.val_ids]
            self.epoch_size = len(self.val_ids)

        else:
            self.data = [self.data[idx] for idx in self.train_ids]

        self.missing_sensors = []
        for k in range(self.resp):
            miss_sens_resp = []
            counter = 0
            for i in range(self.hi):
                for j in range(self.wi):
                    miss_address = [i, j, k]
                    if miss_address not in self.all_channel_address:
                        miss_sens_resp.append(deepcopy(counter))
                    counter += 1
            self.missing_sensors.append(deepcopy(miss_sens_resp))

        self.events = [x for x in range(len(self.data))]
        self.data_arr = []

        self.batch_lengths = []
        self.total_batch_length = 0

        for idx, event in enumerate(self.data):
            self.channels.append(len(event))
            all_len = []
            for ch in range(len(event)):
                all_len.append(len(event[ch]))
            min_len = min(all_len)
            self.points.append(deepcopy(min_len))

            batch_length = (min_len - self.seq_len) #// self.batch_size

            self.total_batch_length += batch_length
            self.batch_lengths.append(deepcopy(self.total_batch_length))

            for ch in range(len(event)):
                del self.data[idx][ch][min_len:]

            event_grid = np.zeros((min_len, self.hi, self.wi, self.resp))
            event_arr = np.array(event).swapaxes(0, 1)

            for ch in range(self.maxch):
                ch_add = self.all_channel_address[ch]
                event_grid[:, ch_add[0], ch_add[1], ch_add[2]] = event_arr[:, ch]
            self.data_arr.append(event_grid)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.total_batch_length

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        access_id = sum(i - 1 < index for i in self.batch_lengths)

        event_id = self.events[access_id]

        if access_id > 0:
            new_index = index - self.batch_lengths[access_id-1]
        else:
            new_index = index

        # print(event_id)
        # Generate data
        data_tuple = self.__data_generation(new_index, event_id)
        return data_tuple

    def on_epoch_end(self):  # Edited by KE
        'Updates indexes after each epoch'
        pass

    def __data_generation(self, index, event_id):  # Edited by KE
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        n_ch = self.channels[event_id]
        seq = np.zeros((self.batch_size, self.n_models, self.past_steps, self.hi, self.wi, self.resp))
        seq2 = np.zeros((self.batch_size, self.n_models, self.past_steps, self.hi, self.wi, self.resp))
        # Generate data
        start = index #*self.batch_size + i
        seq_single = self.data_arr[event_id][start:start + self.past_steps]
        seq = deepcopy(np.repeat(seq_single[np.newaxis], repeats=self.batch_size*self.n_models, axis=0))
        seq = seq.reshape((self.batch_size, self.n_models, self.past_steps, self.hi, self.wi, self.resp))

        start2 = index+self.future_steps #*self.batch_size + i + self.future_steps
        seq_single = self.data_arr[event_id][start2:start2 + self.past_steps]
        seq2 = deepcopy(np.repeat(seq_single[np.newaxis], repeats=self.batch_size*self.n_models, axis=0))
        seq2 = seq2.reshape((self.batch_size, self.n_models, self.past_steps, self.hi, self.wi, self.resp))



        seq = np.moveaxis(seq, 1, -2).reshape((self.batch_size, self.past_steps, self.hi, self.wi, -1))
        seq2 = np.moveaxis(seq2, 1, -2).reshape((self.batch_size, self.past_steps, self.hi, self.wi, -1))


        if self.normalize:
            rms = np.average(np.square(seq))
            seq = seq / np.sqrt(rms)
            rms2 = np.average(np.square(seq2))
            seq2 = seq2 / np.sqrt(rms2)

        return (seq, seq2)