import numpy as np

def ReduceMemoryUsage(arr: np.ndarray, info: bool = True, inplace: bool = False):
    before = arr.nbytes / (1024 ** 3)  # Convert bytes to GB

    # Function to check if a value can fit in the given dtype range
    def is_in_dtype_range(val, dtype):
        return np.iinfo(dtype).min <= val <= np.iinfo(dtype).max

    # Mapping of numpy dtypes to their corresponding downcast dtypes
    downcast_mapping = {
        np.int64: [np.int8, np.int16, np.int32],
        np.int32: [np.int8, np.int16],
        np.int16: [np.int8],
        np.float64: [np.float32]
    }

    # Create a new array with the same shape and data type as the original array
    new_arr = arr.copy()

    # Iterate through each column of the array
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        col_type = col.dtype
        c_min = col.min()
        c_max = col.max()

        # Check if the column data type can be downcasted to a more memory-efficient type
        if col_type in downcast_mapping:
            for downcast_dtype in downcast_mapping[col_type]:
                if is_in_dtype_range(c_min, downcast_dtype) and is_in_dtype_range(c_max, downcast_dtype):
                    new_arr[:, col_idx] = col.astype(downcast_dtype)
                    break

    if info:
        after = new_arr.nbytes / (1024 ** 3)  # Convert bytes to GB
        print(f"Memory usage: {before:.4f} GB => {after:.4f} GB")

    if inplace:
        # If inplace is True, modify the original array with the optimized array
        arr[:, :] = new_arr
    else:
        # If inplace is False, return the optimized array
        return new_arr
