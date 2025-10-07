import torch
import os
import pickle
from torch.utils.data import Dataset

from config import BASE_DIR


class CustomDataset(Dataset):
    """
    Generates and saves large train and test datasets.
    Use "get_data" to get a subset of the saved dataset with given sizes.
    """

    def __init__(self, random_seed, horizon, exp_name: str, file_name: str):
        self.random_seed = random_seed
        self.horizon = horizon
        self._data = None
        torch.manual_seed(self.random_seed)

        # file name and path
        file_path = os.path.join(BASE_DIR, exp_name, 'dataset')
        path_exist = os.path.exists(file_path)
        if not path_exist:
            os.makedirs(file_path)
        self.file_name = os.path.join(file_path, file_name)

    def _generate_data(self, num_samples):
        raise NotImplementedError(
            "You must implement the _generate_data function for CustomDataset." #It must be implemented in the child class
        )

    def get_data(self, num_train_samples=1024, num_valid_samples = 1024, num_test_samples=1024):
        """
        Main function to get train and test datasets.
        """
        self._load_data()
        train_data = self._data['train_data_full'][0:num_train_samples, :, :]
        valid_data = self._data['valid_data'][0:num_valid_samples, :, :]
        test_data = self._data['test_data'][0:num_test_samples, :, :]
        return train_data, valid_data, test_data

    # ---- save and load functions. no need to modify ----
    def _save_data(self):
        """
        Generates and saves train and test datasets.
        """
        train_data_full = self._generate_data(1024)
        valid_data = self._generate_data(1024)
        test_data = self._generate_data(1024)
        # save
        filehandler = open(self.file_name, 'wb')
        pickle.dump({'train_data_full': train_data_full.detach().cpu(),
                     'valid_data': valid_data.detach().cpu(),
                     'test_data': test_data.detach().cpu()},
                    filehandler)
        filehandler.close()

    def _load_data(self):
        """
        Loads the data. If data doesn't exist, saves it.
        """
        # check if data exists
        if not os.path.isfile(self.file_name):
            self._save_data()
        # load data
        filehandler = open(self.file_name, 'rb')
        self._data = pickle.load(filehandler)
        filehandler.close()

    def __len__(self):
        return self._data['train_data_full'].shape[0]

    def __getitem__(self, idx):
        return self._data['train_data_full'][idx, :, :]
