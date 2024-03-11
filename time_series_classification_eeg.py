import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import tensorflow as tf #
from keras.callbacks import ModelCheckpoint 
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

path="/scratch/hareluyaboy/"
file=pd.read_csv(path+"train.csv")
#
#
## check if there is duplicated
#file['label_id'].duplicated().sum()
#
## check which one is duplicated
#file['label_id'][file['label_id'].duplicated()]
#
#
##  one-hot encode or label encode first
#
##one hot encode
## y_enconded=pd.get_dummies(file['expert_consensus'])
#
## label encode
#label_encoder = LabelEncoder()
#y_enconded=label_encoder.fit_transform(file['expert_consensus'])
#file['y_encoded'] = label_encoder.fit_transform(file['expert_consensus'])
#
## read eeg and spectro data
## label_id=train_file.label_id
#label_id=['557980729',
#'1949834128',
#'3790867376',
#'2641122017',
#'1991146353',
#'2660019601',
#'1838186590',
#'545592764',
#'2295678777',
#'4101058765',
#'4257764155']
#
#train_file=file[file['label_id']==int(label_id[0])]
#eeg = pd.read_parquet(path+'train_eegs/'+str(int(train_file['eeg_id']))+'.parquet')
#eeg_offset = int(train_file.eeg_label_offset_seconds)
#train_eeg = eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]
#
## check train data dimention
#train_eeg.shape
#file['expert_consensus'].nunique()
#
#spectrogram = pd.read_parquet(path+'train_spectrograms/'+str(int(train_file['eeg_id']))+'.parquet')
#spec_offset = int(train_file.spectrogram_label_offset_seconds)
#spectrogram = spectrogram.loc[(spectrogram.time>=spec_offset)
#                     &(spectrogram.time<spec_offset+600)]

# eeg_dir="I:/python/time serie classification/train_eegs/"
# DataGenerator 
#label_id=['557980729',
#'1949834128',
#'3790867376',
#'2641122017']


# path="/scratch/hareluyaboy/"

class eegGenerator(Sequence):
    def __init__(self, path, batch_size, label_id, validation=False):
        self.label_id = label_id
        self.batch_size = batch_size
        self.path = path
        self.validation = validation
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.label_id) / self.batch_size)
    
    def __load__(self, one_label_id):

        file=pd.read_csv(path+"train.csv")
        # label encoded
        label_encoder = LabelEncoder()
        # y_enconded=label_encoder.fit_transform(file['expert_consensus'])
        file['y_encoded'] = label_encoder.fit_transform(file['expert_consensus'])
        train_file=file[file['label_id']==int(one_label_id)]
        data = pd.read_parquet(self.path+'train_eegs/'+str(int(train_file['eeg_id'].iloc[0]))+'.parquet')
        eeg_offset = int(train_file.eeg_label_offset_seconds.iloc[0])
        using_window = data.iloc[eeg_offset*200:(eeg_offset+50)*200]
        
         # fill NA by 0
        using_window=using_window.fillna(0)
        data_array = using_window.to_numpy()
        
        # scaler=StandardScaler()
        scaler = MinMaxScaler()
        eeg_train=scaler.fit_transform(data_array)
        
        y=train_file['y_encoded']
        
        return eeg_train,y


    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.label_id):
            self.batch_size = len(self.label_id) - index*self.batch_size
        
        # Select the right batch
        batch_eeg = self.label_id[index*self.batch_size : (index+1)*self.batch_size]

        batch_eeg_data = []
        eeg_labels=[]
        for one_label_id in batch_eeg:
            _eeg_train,_eeg_label=self.__load__(one_label_id)
            batch_eeg_data.append(_eeg_train)
            eeg_labels.append(_eeg_label)

        return np.array(batch_eeg_data), np.array(eeg_labels)  

    def on_epoch_end(self):
        pass


# build model
def model_use(model='CNN'):
    if model=='CNN':
        model = keras.Sequential([
            layers.InputLayer(input_shape=[10000,20]),
            # Data Augmentation （improv model fit）
            # preprocessing.RandomContrast(factor=0.10),
            # preprocessing.RandomFlip(mode='horizontal'),
            # preprocessing.RandomRotation(factor=0.10),
        
            
            # First Convolutional Block
            layers.BatchNormalization(renorm=True),
            layers.Conv1D(filters=128, kernel_size=5, activation="relu", padding='causal'),
            layers.Conv1D(filters=128, kernel_size=5, activation="relu", padding='causal'),
                          # give the input dimensions in the first layer
                          # [height, width, color channels(RGB)]
                          # input_shape=[128, 128, 3]),
            layers.Dropout(0.3),
            layers.MaxPool1D(),
            # layers.GlobalAveragePooling1D(),
            # Second Convolutional Block
            layers.BatchNormalization(renorm=True),
            #layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding='same'),
            layers.Conv1D(filters=128, kernel_size=5, activation="relu", padding='causal'),
            layers.Conv1D(filters=128, kernel_size=5, activation="relu", padding='causal'),
            layers.Dropout(0.3),
            # layers.MaxPool1D(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            # Third Convolutional Block
        #     layers.BatchNormalization(renorm=True),
        #     #layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding='same'),
        #     layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding='causal'),
        #     # may add more Block
        #     # layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
        #     layers.Dropout(0.3),
        #     layers.MaxPool1D(),
        #     layers.GlobalAveragePooling1D(),
        #     # Classifier Head
        #     layers.BatchNormalization(renorm=True),
        #     layers.Flatten(),
        #     layers.Dense(units=128, activation="relu"),
        #     layers.Dropout(0.3),
        #     layers.BatchNormalization(renorm=True),
        #     layers.Dense(units=64, activation="relu"),
        #     layers.Dropout(0.3),
        # #    layers.BatchNormalization(renorm=True),
        # #    layers.Dense(units=32, activation="relu"),
        # #    layers.Dropout(0.5),
        #     # layers.BatchNormalization(renorm=True),
        #     # layers.Dense(units=32, activation="relu"),
        #     # layers.Dropout(0.2),
        #     # layers.Dense(units=2, activation="relu"),
        #     # layers.Dropout(0.3),
        #     # layers.Dense(units=24, activation="relu"),
        #     # layers.Dense(units=24, activation="relu"),
        #     layers.BatchNormalization(renorm=True),
            layers.Dense(units=6, activation="softmax"),
            
        ])
        # model.summary()
    if model=='LSTM':
        model = keras.Sequential([
        layers.InputLayer(input_shape=[10000,20]),
        # Data Augmentation （improv model fit）
        # preprocessing.RandomContrast(factor=0.10),
        # preprocessing.RandomFlip(mode='horizontal'),
        # preprocessing.RandomRotation(factor=0.10),
    
        
        # First Convolutional Block
        layers.BatchNormalization(renorm=True),
        layers.LSTM(128),
        layers.Dropout(0.3),
        layers.BatchNormalization(renorm=True),
        layers.Dense(units=100, activation="relu"),
        layers.Dropout(0.3),
        layers.BatchNormalization(renorm=True),
        layers.Dense(units=100, activation="relu"),
        layers.Dense(units=100, activation="relu"),
        # layers.Dropout(0.3),
        # layers.BatchNormalization(renorm=True),
        # layers.Dense(units=32, activation="relu"),
        # layers.Dropout(0.2),
        # layers.Dense(units=2, activation="relu"),
        # layers.Dropout(0.3),
        # layers.Dense(units=24, activation="relu"),
        # layers.Dense(units=24, activation="relu"),
        layers.Dropout(0.3),
        # layers.BatchNormalization(renorm=True),
        layers.Dense(units=6, activation="softmax"),
        
        ])
    if model=="DNN":
        model = keras.Sequential([
        layers.InputLayer(input_shape=[10000,20]),
        layers.Dense(units=100, activation="relu"),
        layers.Dropout(0.3),
        layers.BatchNormalization(renorm=True),
        layers.Dense(units=100, activation="relu"),
        layers.Dropout(0.3),
        layers.BatchNormalization(renorm=True),
        layers.Dense(units=100, activation="relu"),
        layers.Dropout(0.3),
        layers.BatchNormalization(renorm=True),
        layers.Dense(units=100, activation="relu"),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(units=6, activation="softmax"),
        
        ])
    if model=='CNN_LSTM':
        model = keras.Sequential([
            layers.InputLayer(input_shape=[10000,20]),
            # Data Augmentation （improv model fit）
            # preprocessing.RandomContrast(factor=0.10),
            # preprocessing.RandomFlip(mode='horizontal'),
            # preprocessing.RandomRotation(factor=0.10),
        
            
            # First Convolutional Block
            layers.BatchNormalization(renorm=True),
            layers.Conv1D(filters=128, kernel_size=5, activation="relu", padding='causal'),
            layers.Conv1D(filters=128, kernel_size=5, activation="relu", padding='causal'),
                          # give the input dimensions in the first layer
                          # [height, width, color channels(RGB)]
                          # input_shape=[128, 128, 3]),
            # layers.Dropout(0.3),
            layers.MaxPool1D(),
            # layers.GlobalAveragePooling1D(),
            # Second Convolutional Block
            layers.BatchNormalization(renorm=True),
            #layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding='same'),
            layers.Conv1D(filters=128, kernel_size=5, activation="relu", padding='causal'),
            layers.Conv1D(filters=128, kernel_size=5, activation="relu", padding='causal'),
            # layers.Dropout(0.3),
            # layers.MaxPool1D(),
            layers.GlobalAveragePooling1D(),
            # layers.Dropout(0.3),
            layers.LSTM(128),
            layers.Dropout(0.3),
            layers.BatchNormalization(renorm=True),
            layers.Dense(units=100, activation="relu"),
            layers.Dropout(0.3),
            layers.BatchNormalization(renorm=True),
            layers.Dense(units=100, activation="relu"),
            layers.Dropout(0.3),
            layers.BatchNormalization(renorm=True),
            layers.Dense(units=100, activation="relu"),
            layers.Dropout(0.3),
            layers.BatchNormalization(renorm=True),
            layers.Dense(units=64, activation="relu"),
            # layers.Dropout(0.3),
            # layers.BatchNormalization(renorm=True),
            # layers.Dense(units=32, activation="relu"),
            # layers.Dropout(0.2),
            # layers.Dense(units=2, activation="relu"),
            # layers.Dropout(0.3),
            # layers.Dense(units=24, activation="relu"),
            # layers.Dense(units=24, activation="relu"),
            layers.Dropout(0.3),
            # layers.BatchNormalization(renorm=True),
            layers.Dense(units=6, activation="softmax"),
        
            
        ])
        
    if model=="RNN":
        model = keras.Sequential([
        layers.InputLayer(input_shape=[10000,20]),
        layers.LSTM(128,return_sequences=True),
        # layers.Dropout(0.3),
        layers.BatchNormalization(renorm=True),
        layers.LSTM(128,return_sequences=True),
        layers.BatchNormalization(renorm=True),
        layers.LSTM(128),
        layers.Dense(units=128, activation="relu"),
        layers.Dropout(0.3),
        layers.BatchNormalization(renorm=True),
        layers.Dense(units=80, activation="relu"),
        layers.Dropout(0.3),
        layers.BatchNormalization(renorm=True),
        layers.Dense(units=80, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(units=6, activation="softmax"),
        
        ])
    model.summary()
    return model
   
using_model='CNN'
print('Using '+using_model+' model')
model=model_use(model="CNN")

# add early stop to prevent overfitting
filepath="/home/hareluyaboy/ondemand/machine_learning/time_series_classification/model_{epoch:02d}-{val_accuracy:.2f}.h5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

# Train data
model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)



eeg_train_id, eeg_val_id = train_test_split(file['label_id'], test_size=0.3, random_state=42,shuffle=True)


# check label distribution
eeg_train_counts = file[file['label_id'].isin(eeg_train_id)]['expert_consensus'].value_counts()
eeg_val_counts= file[file['label_id'].isin(eeg_val_id)]['expert_consensus'].value_counts()
print(eeg_train_counts)
print(eeg_val_counts)

eeg_train=eegGenerator(path=path, batch_size=50, label_id=eeg_train_id)
eeg_val=eegGenerator(path=path, batch_size=50, label_id=eeg_val_id)

start_time = time.time()
# train model
history = model.fit(
    eeg_train,
    validation_data=eeg_val,
    epochs=50,
    verbose=1,
    callbacks=[checkpointer],
)


checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)


end_time = time.time()

training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))
# Save the model after training
model.save("/home/hareluyaboy/ondemand/machine_learning/time_series_classification/"+using_model+"_model.h5")
model.save("/home/hareluyaboy/ondemand/machine_learning/time_series_classification/"+using_model+"_model.keras")
# model.save("C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/model.keras")
history_frame = pd.DataFrame(history.history)
# history_frame.loc[:, ['loss', 'val_loss']].plot()
# history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()

history_frame.loc[:, ['loss', 'val_loss']].plot()
plt.savefig('/home/hareluyaboy/ondemand/machine_learning/time_series_classification/'+using_model+'_Loss.png',dpi=600)
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.savefig('/home/hareluyaboy/ondemand/machine_learning/time_series_classification/'+using_model+'_Accuracy.png',dpi=600)


# # test
# def load(one_label_id):

#     file=pd.read_csv(path+"train.csv")
#     # label encoded
#     label_encoder = LabelEncoder()
#     # y_enconded=label_encoder.fit_transform(file['expert_consensus'])
#     file['y_encoded'] = label_encoder.fit_transform(file['expert_consensus'])
#     train_file=file[file['label_id']==int(one_label_id)]
#     data = pd.read_parquet(path+'train_eegs/'+str(int(train_file['eeg_id']))+'.parquet')
#     eeg_offset = int(train_file.eeg_label_offset_seconds)
#     using_window = data.iloc[eeg_offset*200:(eeg_offset+50)*200]
    
#       # fill NA by 0
#     using_window=using_window.fillna(0)
#     data_array = using_window.to_numpy()
    
#     scaler=StandardScaler()

#     eeg_train=scaler.fit_transform(data_array)
    
#     y=train_file['y_encoded']
    
#     return eeg_train,y


# batch_eeg = label_id[0:2]
# batch_eeg_data = []
# eeg_labels=[]
# for one_label_id in batch_eeg:
#     _eeg_train,_eeg_label=load(one_label_id)
#     batch_eeg_data.append(_eeg_train)
#     eeg_labels.append(_eeg_label)
    
# np.array(batch_eeg_data).shape
# np.array(eeg_labels).shape

