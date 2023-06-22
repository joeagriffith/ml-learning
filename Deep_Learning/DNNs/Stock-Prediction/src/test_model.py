import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


from preprocess import preprocess_df, generate_sequences, normalize_sequences

SEQ_LEN = 500
TARGET = "max_pct"

# reconstructed = tf.keras.models.load_model("../models/BTCUSDT-5HR_HIGH-250-SEQ-1641157333")
reconstructed = tf.keras.models.load_model("../models/RNN_Final-32-0.031.model")

#IMPORT DATASET FROM CSV INTO PANDAS.DATAFRAME
main_df = pd.DataFrame()
main_df = pd.read_csv("../data/Binance_BTCUSDT_1h_price.csv", names=["index", "high", "low", "close", "volume", "max_pct", "min_pct"])


main_df.rename(columns={f"{TARGET}": "target"}, inplace=True)

#INDEX EACH SAMPLE BY UNIX
main_df.set_index("index", inplace=True)
main_df = main_df[["high", "low", "close", "volume", "target"]]

#SPLIT DATAFRAME INTO TRAIN AND VALIDATION SET
times = sorted(main_df.index.values)
last_5pct = times[-int(0.1*len(times))]
test_df = main_df[(main_df.index >= last_5pct)]



#CHANGE DF INTO INPUT AND TARGET SEQUENCES
test_x, test_y = generate_sequences(test_df, SEQ_LEN)

#NORMALIZE EACH SEQUENCE BY MIN/MAX PRICE AND MIN/MAX VOLUME
means = []

for i in range(len(test_y)):
    means.append(np.mean(test_x[i,:,:-1]))
test_x, test_y = normalize_sequences(test_x, test_y)

predict = []
for i in range(len(test_y)):
    predict.append(reconstructed(np.array([test_x[i]])) * means[i])
predict_array = np.array(predict)
plt.plot(range(len(test_y)), predict_array[:,0,0], color='red')
plt.plot(range(len(test_y)), np.multiply(test_y, means), color='orange')
plt.plot(range(len(test_y)), np.array(test_df)[-len(test_y):,2])
plt.show()