import time
import psutil
import os

def memory_usage_psutil():
    # Return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem

def check_memory_and_time(func):
    def wrapper(*args, **kwargs):
        # Record the starting time and memory
        start_time = time.time()
        start_mem = memory_usage_psutil()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Record the ending time and memory
        end_time = time.time()
        end_mem = memory_usage_psutil()
        
        # Calculate the differences
        time_taken = end_time - start_time
        memory_used = end_mem - start_mem
        
        print(f"Function '{func.__name__}' took {time_taken:.4f} seconds to run.")
        print(f"Memory usage: {memory_used:.2f} MB")
        
        return result
    return wrapper