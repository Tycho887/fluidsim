import logging
import numpy as np
import time

def timer_decorator(func):
    """
    Decorator that times the execution of a function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__}\nExecution time: {execution_time:.3f} s\n{'-'*50}")
        return result
    return wrapper

def logging_decorator(func):
    """
    Decorator that logs the execution of a function
    """
    def wrapper(*args, **kwargs):
        logging.info(f"Running {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Finished {func.__name__}")
        return result
    return wrapper

def logging_and_timing_decorator(func):
    """
    Decorator that logs and times the execution of a function
    """
    def wrapper(*args, **kwargs):

        start_time = time.time()

        logging.info(f"Running {func.__name__}")

        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"{func.__name__}\nExecution time: {execution_time:.3f} s\n{'-'*50}")

        logging.info(f"Finished {func.__name__}")
        return result
    return wrapper