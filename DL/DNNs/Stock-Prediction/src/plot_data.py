import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocess import preprocess_df, generate_sequences, normalize_sequences 

SEQ_LEN = 500
TARGET = "max_high"


#IMPORT DATASET FROM CSV INTO PANDAS.DATAFRAME
main_df = pd.DataFrame()
main_df = pd.read_csv("../data/Binance_BTCUSDT_1h_price.csv", names=["index", "high", "low", "close", "volume", "max_high", "min_low"])
main_df.rename(columns={f"{TARGET}": "target"}, inplace=True)

#INDEX EACH SAMPLE BY UNIX
main_df.set_index("index", inplace=True)
main_df = main_df[["high", "low", "close", "volume", "target"]]

# #SPLIT DATAFRAME INTO TRAIN AND VALIDATION SET
# times = sorted(main_df.index.values)
# last_5pct = times[-int(0.05*len(times))]
# validation_main_df = main_df[(main_df.index >= last_5pct)]
# main_df = main_df[(main_df.index < last_5pct)]

# #PREPROCESS THE DATAFRAMES
train_x, train_y = generate_sequences(main_df, SEQ_LEN)
train_x, train_y = normalize_sequences(train_x, train_y)
# validation_x, validation_y = preprocess_df(validation_main_df, SEQ_LEN)

train_xpoints = np.array(range(len(train_x[0,:,0])))
train_ypoints = train_x[1000,:,3]
plt.plot(train_xpoints, train_ypoints)
plt.show()

# validation_xpoints = np.array(range(len(validation_y)))
# validation_ypoints = validation_y
# plt.plot(validation_xpoints, validation_ypoints)
# plt.show()
