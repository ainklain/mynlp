
from timeseries.config import Config
from timeseries.model import TSModel
from timeseries.data_process import dataset_process, load_data, DataGenerator, DataScheduler


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

configs = Config()

# get data for all assets and dates
ds = DataScheduler(configs)
ds.test_end_idx = ds.base_idx + 1000

ii = 0
# while not ds.done:
for _ in range(1):
    model = TSModel(configs)
    ds.set_idx(5000)
    ds.train(model,
             train_steps=configs.train_steps,
             eval_steps=10,
             save_steps=200,
             early_stopping_count=20,
             model_name=configs.f_name)




    test_dataset_list, features_list = ds.test(model)

    ds.next()