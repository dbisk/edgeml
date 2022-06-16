"""tfmemtest.py

Memory usage experimentation using the SuDoRM-RF model found in `models`, using
TensorFlow.

There may be certain issues with combined-memory devices, such as the NVIDIA
Jetson Nano and other SBCs with integrated CPU + graphics.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

import tensorflow as tf
from tqdm import tqdm

# script should not be imported
if __name__ != '__main__':
    raise RuntimeError("This is a script and should not be imported!")
    exit(1)

# parameters for memory testing
NUM_LOOPS = 10
BATCH_SIZE = 1

# identify if we have a GPU
USING_GPU = True if len(tf.config.list_physical_devices('GPU')) > 0 else False
print(f'Using GPU? {USING_GPU}.')

if USING_GPU:
    # evaluate the gpu cache
    pass
else:
    import psutil
    base_mem = psutil.virtual_memory()[3] / 1.0e9

# initialize the model
model = tf.keras.applications.MobileNetV3Small()

# define forward and backward pass functions to call in testing
def forward_pass(model, x):
    pass

def backward_pass(model, x, y):
    pass

# estimate the amount of memory used for just the model
if USING_GPU:
    tf.config.experimental.get_memory_info('GPU:0')
else:
    pre_loop_base_mem = psutil.virtual_memory()[3] / 1.0e9 - base_mem
    print(f"Model-only CPU memory usage: {pre_loop_base_mem}")

# perform memory testing
peak_mems = []
for i in tqdm(range(NUM_LOOPS)):
    # reset the baseline memory allocated
    if USING_GPU:
        tf.config.experimental.reset_memory_stats('GPU:0')
    
    # prepare fake data. we don't need to use any real data for memory profiling
    # since we don't actually care about the model performance.
    dummy_input = tf.random.uniform([BATCH_SIZE, 244, 244, 3])

    # TODO


