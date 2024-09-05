"""
Testing script for DCEM
"""
from run import DCEM_run
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    DCEM_run(model_name='dcem', dataset_name='mosi', is_tune=False, seeds=[1112], model_save_dir="./pt",
             res_save_dir="./result", log_dir="./log", mode='test', is_distill=False)
