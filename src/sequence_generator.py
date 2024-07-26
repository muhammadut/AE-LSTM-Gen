import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from typing import List, Tuple, Optional, Union


class SequenceGenerator:
    """
    A class for generating sequences from a DataFrame.

    This class handles the conversion of DataFrame rows into sequences,
    optionally including a target variable.

    Attributes:
        df (pd.DataFrame): The input DataFrame.
        id_column (str): Name of the column containing unique identifiers.
        sequence_column (str): Name of the column used to order the sequence.
        target_column (Optional[str]): Name of the target variable column, if any.
    """

    def __init__(self, df: pd.DataFrame, id_column: str, sequence_column: str, 
                 target_column: Optional[str] = None):
        """
        Initialize the SequenceGenerator.

        Args:
            df (pd.DataFrame): The input DataFrame.
            id_column (str): Name of the column containing unique identifiers.
            sequence_column (str): Name of the column used to order the sequence.
            target_column (Optional[str], optional): Name of the target variable column. Defaults to None.
        """
        self.df = df
        self.id_column = id_column
        self.sequence_column = sequence_column
        self.target_column = target_column
    
    def generate_sequences(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate sequences from the DataFrame.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                If target_column is None, returns only the padded sequences.
                If target_column is specified, returns a tuple of (padded_sequences, labels).
        """
        sequences: List[np.ndarray] = []
        labels: List[Union[int, float]] = []
        unique_ids = self.df[self.id_column].unique()

        for unique_id in tqdm(unique_ids, desc="Generating sequences"):
            data = self.df[self.df[self.id_column] == unique_id]
            drop_columns = [self.id_column, self.sequence_column]
            if self.target_column:
                drop_columns.append(self.target_column)
            sequence = data.drop(columns=drop_columns).values
            sequences.append(sequence)
            if self.target_column:
                label = data[self.target_column].iloc[-1]
                labels.append(label)

        padded_sequences = pad_sequences(sequences, padding='post', dtype='float')
        
        if self.target_column:
            return np.array(padded_sequences), np.array(labels)
        
        return np.array(padded_sequences)