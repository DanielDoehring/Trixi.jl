import h5py
import numpy as np

# Open the file in read mode
with h5py.File('CF_x_000000.h5', 'r') as f:
  # List all groups
  print("Keys: %s" % f.keys())
  
  # Iterate over all keys
  for key in f.keys():
    data = f[key]

    data_array = np.array(data)
    if np.ndim(data_array) > 1:
      # Transpose to get dimensions right
      data_array = data_array.T

    print("Dimensions of array for key %s: %s" % (key, data_array.shape))
    print("Data for key %s: %s" % (key, data_array))