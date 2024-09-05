"""
ATIO -- All Trains in One
"""
from .singleTask import *
from .singleTask import DCEM
__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'dcem': DCEM
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
