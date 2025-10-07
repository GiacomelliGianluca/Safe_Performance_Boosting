import pickle
import os

from config import BASE_DIR


def saving_data(data_dic, f_name: str):
    """
    Save results.
    """
    # file name and path
    file_path = os.path.join(BASE_DIR, 'experiments/tank/saved_results')
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_name = os.path.join(file_path, f_name)
    # save
    filehandler = open(file_name, 'wb')
    with filehandler as file:
        pickle.dump(data_dic, file)
    filehandler.close()

