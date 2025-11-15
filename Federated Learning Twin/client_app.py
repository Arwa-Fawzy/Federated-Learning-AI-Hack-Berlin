import flwr as fl
import tensorflow as tf
import numpy as np
from client_utils import load_and_prepare_sequences

def build_seq_autoencoder(input_shape, latent_dim=64):
    """
    Simple sequence autoencoder using TimeDistributed Dense over flattened timesteps.
    Encoder returns latent vector; decoder reconstructs sequences.
    """
    seq_len, feat = input_shape
    inp = tf.keras.layers.Input(shape=(seq_len, feat))
    
    # Encoder
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu"))(inp)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2))(x)
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)
    
    # Decoder
    d = tf.keras.layers.Dense(seq_len * 128, activation="relu")(z)
    d = tf.keras.layers.Reshape((seq_len, 128))(d)
    d = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(feat, activation="linear"))(d)
    
    model = tf.keras.Model(inputs=inp, outputs=d)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


class DriftTwinClient(fl.client.NumPyClient):
    def __init__(self, cid: int, epochs: int = 5, batch_size: int = 32, seq_len: int = 50):
        self.cid = cid
        self.epochs = epochs
        self.batch_size = batch_size
        
        print(f"[client {cid}] loading data and preparing sequences...")
        (self.x_train, self.x_hold, self.x_test, self.y_test,
         self.input_shape, self.scaler, self.meta) = load_and_prepare_sequences(cid, seq_len=seq_len)
        
        print(f"[client {cid}] seqs: train={len(self.x_train)}, hold={len(self.x_hold)}, "
              f"test={len(self.x_test)}, features={self.input_shape[1]}")
        
        self.model = build_seq_autoencoder(self.input_shape, latent_dim=64)
        self.local_threshold = None  # computed later from hold set

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        if len(self.x_train) == 0:
            return self.model.get_weights(), 0, {}
        
        self.model.fit(self.x_train, self.x_train,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=0)
        
        # compute local threshold
        data_for_threshold = self.x_hold if len(self.x_hold) > 0 else self.x_train
        recon = self.model.predict(data_for_threshold, verbose=0)
        errs = np.mean((recon - data_for_threshold) ** 2, axis=(1,2))
        self.local_threshold = float(errs.mean() + 3.0 * errs.std())
        
        return self.model.get_weights(), len(self.x_train), {"local_threshold": self.local_threshold}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        if len(self.x_test) == 0:
            return float("nan"), 0, {"accuracy": 0.0}
        
        recon = self.model.predict(self.x_test, verbose=0)
        errs = np.mean((recon - self.x_test) ** 2, axis=(1,2))
        
        threshold = self.local_threshold if self.local_threshold is not None else float(errs.mean() + 3.0 * errs.std())
        preds = (errs > threshold).astype(int)
        y_true = self.y_test.astype(int)
        
        correct = (preds == y_true).sum()
        acc = float(correct) / len(y_true)
        
        return float(errs.mean()), len(y_true), {"accuracy": acc, "threshold": float(threshold)}


# Helper: create a client instance for a given CID
def make_client_instance(cid, epochs=5, batch_size=32, seq_len=50):
    return DriftTwinClient(cid=cid, epochs=epochs, batch_size=batch_size, seq_len=seq_len)
