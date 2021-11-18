from torch import cuda
import os

data_index = 2
data_dirs = ['/opt/ml/input/data/ICDAR17_Korean', '/opt/ml/input/data/dataset_revised', '/opt/ml/input/data/merge_dataset']
train_datasets = ['train', 'annotation', 'train_1']
valid_datasets = ['','','valid_1']

my_config = {
    "program": "main.py",
    
    # "method": "bayes",
    # "metric":{
    #     "name": "loss",
    #     "goal": "minimize"
    # },

    "method": "grid",

    "name": "OCR_Sweep",
    "parameters":
    {
        "seed":
            {"value": 15},
        "data_dir":
            {"value": os.environ.get('SM_CHANNEL_TRAIN', data_dirs[data_index])},
        "train_dataset":
            {"value": train_datasets[data_index]},
        "valid_dataset":
            {"value": valid_datasets[data_index]},
        "dataset_shuffle":
            {"value": True}, # True
        "model_dir":
            {"value": os.environ.get('SM_MODEL_DIR', 'trained_models')},
        "device":
            {"value": 'cuda' if cuda.is_available() else 'cpu'},
        "num_workers":
            {"value": 8},  # 8
        "image_size":
            {"value": 1024},
        "input_size":
            {"value": 512},
        "batch_size":
            {"value": 32},  # 32
        "optimizer":
            {"values": ['adam']}, # ['sgd', 'asgd', 'momentum', 'adam', 'adamax', 'adamw', 'nadam', 'radam']
        # "learning_rate":
        #     {"min": 1e-6, "max": 1e-2},
        "learning_rate":
            {"value": 1e-3},
        "max_epoch":
            {"value": 200},
        "save_interval":
            {"value": 1},
        "name":
            {"value": 'OCR_sweep'},
    }
}