import os
import logging
import sys

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_path)
sys.path.append(project_path)


def make_logger(log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = make_one_logger(log_path, 'logger')
    train_logger = make_one_logger(log_path, 'train_logger')
    val_logger = make_one_logger(log_path, 'val_logger')
    test_logger = make_one_logger(log_path, 'test_logger')
    return logger, train_logger, val_logger, test_logger


def make_one_logger(log_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler_2 = logging.FileHandler(os.path.join(log_path, 'output.log'))
    handler_2.setLevel(logging.DEBUG)
    handler_2.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(handler_2)
    return logger


def make_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if os.path.exists(os.path.join(path, 'output.log')):
            with open(os.path.join(path, 'output.log'), "r+") as f:
                f.seek(0)
                f.truncate()
    return path
