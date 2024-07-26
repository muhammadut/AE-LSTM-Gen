import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, TimeDistributed, Dropout, RepeatVector
from tensorflow.keras import optimizers, losses, regularizers
from tensorflow.keras.callbacks import EarlyStopping, History
import faiss
from typing import Tuple

class GeneralLSTMAutoencoder:
    """
    A general-purpose LSTM Autoencoder for sequence modeling.

    This class implements an LSTM-based autoencoder that can be used for various
    sequence modeling tasks, including anomaly detection and feature extraction.

    Attributes:
        max_sequence_length (int): Maximum length of input sequences.
        num_features (int): Number of features in each timestep of the sequence.
        lstm_units (int): Number of units in LSTM layers.
        dropout_rate (float): Dropout rate for regularization.
        l2_reg (float): L2 regularization factor.
        learning_rate (float): Learning rate for the Adam optimizer.
        model (Sequential): The Keras Sequential model representing the autoencoder.
    """

    def __init__(self, max_sequence_length: int, num_features: int, lstm_units: int = 50, 
                 dropout_rate: float = 0.2, l2_reg: float = 0.001, learning_rate: float = 1e-3):
        """
        Initialize the GeneralLSTMAutoencoder.

        Args:
            max_sequence_length (int): Maximum length of input sequences.
            num_features (int): Number of features in each timestep of the sequence.
            lstm_units (int, optional): Number of units in LSTM layers. Defaults to 50.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
            l2_reg (float, optional): L2 regularization factor. Defaults to 0.001.
            learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-3.
        """
        self.max_sequence_length = max_sequence_length
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """
        Build and compile the LSTM autoencoder model.

        Returns:
            Sequential: Compiled Keras Sequential model.
        """
        model = Sequential([
            Masking(mask_value=0., input_shape=(self.max_sequence_length, self.num_features)),
            LSTM(self.lstm_units, activation='relu', return_sequences=False, 
                 kernel_regularizer=regularizers.l2(self.l2_reg)),
            Dropout(self.dropout_rate),
            RepeatVector(self.max_sequence_length),
            LSTM(self.lstm_units, activation='relu', return_sequences=True, 
                 kernel_regularizer=regularizers.l2(self.l2_reg)),
            Dropout(self.dropout_rate),
            TimeDistributed(Dense(self.num_features))
        ])
        
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), 
                      loss=losses.MeanSquaredError())
        return model
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 128, 
            validation_split: float = 0.2, patience: int = 10) -> History:
        """
        Fit the model to the input data.

        Args:
            X (np.ndarray): Input sequences.
            epochs (int, optional): Number of epochs to train. Defaults to 100.
            batch_size (int, optional): Batch size for training. Defaults to 128.
            validation_split (float, optional): Fraction of data to use for validation. Defaults to 0.2.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 10.

        Returns:
            History: Training history.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, 
                                       restore_best_weights=True)
        history = self.model.fit(X, X, epochs=epochs, batch_size=batch_size, 
                                 validation_split=validation_split, 
                                 callbacks=[early_stopping])
        return history
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode input sequences using the trained encoder part of the autoencoder.

        Args:
            X (np.ndarray): Input sequences to encode.

        Returns:
            np.ndarray: Encoded representations of input sequences.
        """
        encoder = Sequential(self.model.layers[:2])
        return encoder.predict(X)
    
    @staticmethod
    def normalize_L2(data: np.ndarray) -> np.ndarray:
        """
        Perform L2 normalization on the input data.

        Args:
            data (np.ndarray): Input data to normalize.

        Returns:
            np.ndarray: L2 normalized data.
        """
        faiss.normalize_L2(data)
        return data
    
    def find_nearest_neighbors(self, train_embeddings: np.ndarray, test_embeddings: np.ndarray, 
                               k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k-nearest neighbors for test embeddings in train embeddings.

        Args:
            train_embeddings (np.ndarray): Embeddings of training data.
            test_embeddings (np.ndarray): Embeddings of test data.
            k (int, optional): Number of nearest neighbors to find. Defaults to 8.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of k-nearest neighbors.
        """
        train_embeddings_norm = self.normalize_L2(train_embeddings.astype('float32'))
        test_embeddings_norm = self.normalize_L2(test_embeddings.astype('float32'))
        
        dim = train_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(train_embeddings_norm)
        
        D, I = index.search(test_embeddings_norm, k)
        return D, I

    def summary(self) -> None:
        """
        Print a summary of the model architecture.
        """
        self.model.summary()