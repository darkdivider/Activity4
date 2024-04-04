from config import config
import utils
import os

if __name__=='__main__':
    for d in config.dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    utils.train_model(config)