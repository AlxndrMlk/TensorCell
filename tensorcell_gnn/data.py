
import os

import numpy as np

from spektral.data import Dataset, Graph


class TensorcellDataset(Dataset):
    
    """A Tensorcell dataset."""
    
    def __init__(self, dataset_variant, circular_mapping=False, add_constant_feature=False, add_one_hot_index=False, **kwargs):
        """
        :param dataset_variant: A dataset to pick. Currently takes: `ochota_100k`, `centrum_100k`, `mokotow_100k`
        :type dataset_variant: str
        :param circular_mapping: If node values should be mapped to a unit circle
        :type circular_dataset: bool

        ...
        :return: None
        :rtype: None
        """

        self.dataset_variant = dataset_variant
        self.circular_mapping = circular_mapping
        self.add_constant_feature = add_constant_feature
        self.add_one_hot_index = add_one_hot_index
        
        # Construct filenames
        dataset_info = dataset_variant.split('_')
        district = dataset_info[0]
        n_rows = dataset_info[1]
        
        self.filename_A = f'{district}_A.txt'
        self.filename_Xy = f'{district}_X_{n_rows}.txt'

        super().__init__(**kwargs)


    def read(self):
        
        """
        :return: output
        :rtype: list
        """
        
        # We must return a list of Graph objects
        output = []
        
        # Read files
        adjacency_matrix = np.loadtxt(os.path.join(self.path, self.filename_A))
        features = np.loadtxt(os.path.join(self.path, self.filename_Xy), delimiter=',')
        
        # Construct graph objects
        for row in range(features.shape[0]):

            # If `circular_mapping` -> map to a circular representation
            if self.circular_mapping:
                x = self.get_circular_components(features[row, :-1]).T
            else:
                x = features[row, :-1][:, np.newaxis]

            # Add constant feature 1
            if self.add_constant_feature:
                x = np.hstack([x, np.ones(x.shape[0])[:, np.newaxis]])

            # Add one-hot encoded node label
            if self.add_one_hot_index:

                x_plus_oh = []

                for i, d in enumerate(x):
                    one_hot_index = np.zeros(x.shape[0])
                    one_hot_index[i] = 1
                    x_plus_oh.append(np.hstack([d, one_hot_index]))

                x = np.array(x_plus_oh)

            # Construct a graph 
            output.append(
                Graph(
                    x=x, 
                    a=adjacency_matrix, 
                    y=features[row, -1])
            )

        return output


    def get_circular_components(self, x, period=120):

        """
        Takes a 1D circular variable and returns it's 2D embedding on a unit circle
        where the cos component represents Cartesian x-coordianate and sin component represents Cartesian y-coordinate.
        
        :param x: 
        :type x: int | float

        :return: An array with two 2D circular mapping.
        :rtype: numpy.ndarray
        """
        
        omega = (2*np.pi) / period
        cos_component = np.cos(x*omega)
        sin_component = np.sin(x*omega)

        return np.array([sin_component, cos_component])



def train_dev_test_split(dataset, train_size=.8, dev_size=None):
    """Creates a train-dev-test split of a Spektral dataset"""
    
    # Infer dev size and/or test size
    if not dev_size:
        dev_size = test_size = (1 - train_size) / 2
    else:
        test_size = 1 - train_size - dev_size    
        
    # Check if weights sum up to 1
    assert all(np.array([train_size, dev_size, test_size]) > 0), 'Weights do not sum up to 1.'
    
    print(f'--- SPLIT WEIGHTS ---\nTrain: {train_size:0.3f}\nDev: {dev_size:0.3f}\nTest: {test_size:0.3f}')
    
    # Get indices
    idxs = np.random.permutation(len(dataset))
    
    # Get split positions
    train_split = int(train_size * len(dataset))
    dev_split = int((train_size + dev_size) * len(dataset))
    
    # Get indices
    idx_train, idx_dev, idx_test = np.split(idxs, [train_split, dev_split])
    
    # Split and return
    return dataset[idx_train], dataset[idx_dev], dataset[idx_test]
    
    