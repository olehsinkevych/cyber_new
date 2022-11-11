from pickle import dump
import pandas as pd
import math
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class UniTransformer:
    """Represents data preparation for the RNN based model.

    Attributes:
        sequence (pd.DadaFrame): uploaded resampled
            time series.
        path (str): path to .csv with the data.
        scaler (sklearn.preprocessing object): scaler used to scale
            sequence.
        train (pd.DataFrame): training dataset.
        test (pd.DataFrame): test dataset.
        train_scaled (pd.DataFrame): scaled training dataset.
        test_scaled (pd.DataFrame): scaled test dataset.

    Methods:
        upload(path: str, sample): uploads file from path.
        set_month_range(months): provides sub-frame based on months.
        train_test_split(train_percent):
            splits sequence to train and test (validation) sets.
        scale(scaling): scales sequence to [0, 1] range ('norm').
        transform(data_type, n_in, n_out, dropnan):
            converts train and test sequences to classification problem.
        model_input_reshape(x1, x2): reshapes 2D arrays to 3D arrays.
        save_scaler(path): saves scaler to the provided file path.

    """

    def __init__(self, path: str) -> None:
        """Initializes UniTransformer.

        Args:
            path (str): path to .csv with the data.

        Returns:
            None.
        """

        self.sequence = self.upload(path)
        self.scaler = None
        self.train = None
        self.test = None
        self.train_scaled = None
        self.test_scaled = None

    @staticmethod
    def upload(path: str, sample='60min') -> pd.DataFrame:
        """Uploads file from path. Univariate time series
        should be provided with two columns: date and values.
        Sets column names to 'time' and 'value'.

        Args:
            path (str): path to .csv with the data.
            sample (str): represents sampling period.

        Returns:
            data (pd.DataFrame): read resampled data frame.
        """

        df = pd.read_csv(path)
        df.drop(['Unnamed: 0', 'user', 'tag', 'id'], inplace=True, axis=1)
        df['time_point'] = pd.to_datetime(df['time_point'])
        df.set_index("time_point", inplace=True)

        # resample xx minutes data to 1 hour
        df = df.resample(sample).mean()

        return df

    def set_month_range(self, months=(6, 7, 8)) -> None:
        """Provides sub-frame based on months.

        Args:
            months (List[int]): path to .csv with the data.

        Returns:
            None.
        """

        self.sequence = self.sequence[
            self.sequence.index.month.isin(months)]

    def train_test_split(self, train_percent=0.8) -> None:
        """Splits sequence to train and test (validation) sets.

        Args:
            train_percent (float): percent of data goes to the
                training set.

        Returns:
            None.
        """

        training_length = math.ceil(len(self.sequence) *
                                    train_percent)
        self.train = self.sequence[0:training_length]
        self.test = self.sequence[training_length:]

    def scale(self, scaling='norm') -> None:
        """Scales sequence to [0, 1] range ('norm')
        or zero mean and 1 standard deviation.
        Depends on sklearn.preprocessing module.

        Args:
            scaling (str): scaling method.
            'norm' - normalization
            'standard': standardization.

        Returns:
            None.
        """
        if scaling == 'norm':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaling == 'standard':
            self.scaler = StandardScaler()

        if self.train is not None and self.test is not None:
            self.scaler.fit(self.train)
            # remember to fit scaler on the training data firstly!
            self.train_scaled = pd.DataFrame(
                data=self.scaler.transform(self.train),
                index=self.train.index, columns=['value'])
            self.test_scaled = pd.DataFrame(
                data=self.scaler.transform(self.test),
                index=self.test.index, columns=['value'])
        else:
            raise ValueError('train_test_split must be invoked firstly')

    def transform(self, data_type='train', n_in=24, n_out=24,
                  dropnan=True) -> Tuple[np.array, np.array]:
        """Converts train and test sequences to classification problem.
        It is the legacy method. Applied only to the scaled data.

        Args:
            data_type (str): specifies which data to transform.
                'train' or 'test' are available options.
            n_in (int): lag,  number of point used to
                predict next sub-sequence.
            n_out (int): number of point to be predicted.
            dropnan (bool): defines droping of nan contained
                sub-sequences.

        Returns:
            x (numpy.array): training data
            y (numpy.array): test data.
        """

        if data_type == 'train':
            data = self.train_scaled
        elif data_type == 'test':
            data = self.test_scaled

        variables = data.shape[1]
        columns = []
        column_names = []

        # foregoing sequence (x-n, ..., x-1)
        for i in range(n_in, 0, -1):
            columns.append(data.shift(i))
            column_names += [('value%d(x-%d)' % (j + 1, i))
                             for j in range(variables)]

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            columns.append(data.shift(-i))
            if i == 0:
                column_names += [('value%d(x)' % (j + 1))
                                 for j in range(variables)]
            else:
                column_names += [('value%d(x+%d)' % (j + 1, i))
                                 for j in range(variables)]

        # concat everything into one data frame
        df = pd.concat(columns, axis=1)
        df.columns = column_names

        # get rid of nan's
        if dropnan:
            df.dropna(inplace=True)

        # convert data frames into the numpy arrays
        x = df.iloc[:, :n_in].to_numpy()
        y = df.iloc[:, :-n_out].to_numpy()
        print(f'X shape: {x.shape}')
        print(f'y shape: {y.shape}')

        return x, y

    def save_scaler(self, path='scaler.pkl') -> None:
        """Saves scaler to the provided file path.

        Args:
            path (str): path to scaler to be saved.

        Returns:
            None.
        """

        dump(self.scaler, open(path, 'wb'))
        print(f'Scaler has been saved to {path}')


def model_input_reshape(x1: np.array,
                        x2: np.array) -> Tuple[np.array, np.array]:
    """Reshapes 2D arrays to 3D arrays.
    Remember, LSTM wants input in [sample, step, feature] form.

    Args:
        x1 (numpy.array): train numpy array.
        x2 (numpy.array): test (validation) numpy array.

    Returns:
        x1_new (numpy.array): train reshaped 3D array.
        x2_new (numpy.array): test reshaped 3D array.
    """

    x1_new = np.reshape(x1, (x1.shape[0], x1.shape[1], 1))
    x2_new = np.reshape(x2, (x2.shape[0], x2.shape[1], 1))

    print(f'x1 new shape: {x1_new.shape}')
    print(f'x2 new shape: {x2_new.shape}')

    return x1_new, x2_new
