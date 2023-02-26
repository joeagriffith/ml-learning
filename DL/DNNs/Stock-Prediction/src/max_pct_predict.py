import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Attention
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot

from preprocess import preprocess_df, generate_sequences, normalize_sequences

SEQ_LEN = 500
EPOCHS = 100
BATCH_SIZE = 64
LOSS_FUNCTION = 'mean_absolute_percentage_error'
NAME = f"BTCUSDT-50HR_HIGH-{SEQ_LEN}-SEQ-{LOSS_FUNCTION}-LOSS-{int(time.time())}"
TARGET = "max_pct"


#IMPORT DATASET FROM CSV INTO PANDAS.DATAFRAME
main_df = pd.DataFrame()
main_df = pd.read_csv("../data/Binance_BTCUSDT_1h_price.csv", names=["index", "high", "low", "close", "volume", "max_pct", "min_pct"])


main_df.rename(columns={f"{TARGET}": "target"}, inplace=True)

#INDEX EACH SAMPLE BY UNIX
main_df.set_index("index", inplace=True)
main_df = main_df[["high", "low", "close", "volume", "target"]]

#SPLIT DATAFRAME INTO TRAIN AND VALIDATION SET
times = sorted(main_df.index.values)
last_15pct = times[-int(0.15*len(times))]
validation_main_df = main_df[(main_df.index >= last_15pct)]
main_df = main_df[(main_df.index < last_15pct)]


#CHANGE DF INTO INPUT AND TARGET SEQUENCES
train_x, train_y = generate_sequences(main_df, SEQ_LEN)
validation_x, validation_y = generate_sequences(validation_main_df, SEQ_LEN)

#NORMALIZE EACH SEQUENCE BY MIN/MAX PRICE AND MIN/MAX VOLUME
train_x, train_y = normalize_sequences(train_x, train_y)
validation_x, validation_y = normalize_sequences(validation_x, validation_y)

########################################################################################

#BUILD THE MODEL
model = Sequential()


# model.add(Attention())

model.add(LSTM(256, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1))

# MODEL OPTIONS
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-7)

tf.keras.backend.set_epsilon(1)

model.compile(
    loss=LOSS_FUNCTION,
    optimizer=opt,
    metrics=[tf.keras.metrics.MeanAbsoluteError()])

tensorboard = TensorBoard(log_dir=f'../logs/{NAME}')


filepath = "RNN_Final-{epoch:02d}-{val_mean_absolute_error:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("../models/{}.model".format(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

#RUN THE MODEL
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint]
)

#########################################################

#SCORE THE MODEL
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss: ', score[0])
print('Absolute error: ', score[1])
model.save("../models/{}".format(NAME))

