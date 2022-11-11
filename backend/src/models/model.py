"""Contains recurrent NN model for temperature prediction."""

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam


def lstm_model(configs: dict) -> Model:
    """Defines basic LSTM model"""

    inputs = Input(shape=(configs["n_in"], 1))
    layer = LSTM(configs["hidden_vector"],
                 activation=configs["activation"],
                 input_shape=(configs["n_in"], 1))(inputs)
    outputs = Dense(configs["n_out"])(layer)
    model = Model(inputs=inputs, outputs=outputs)
    opt = Adam(learning_rate=configs["learning_rate"])
    model.compile(loss='mse', optimizer=opt)
    return model
