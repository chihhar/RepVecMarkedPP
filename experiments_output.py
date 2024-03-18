import os
import pickle


def open_file(file_path):
    with open(f'{file_path}', 'rb') as pfile:
        file_value = pickle.load(pfile)
    return file_value

def data_open_file(gene,method_name):
    path=pickled/proposed
    if method_name=="early__":
        open_file(f"/{gene}/{method_name}/")