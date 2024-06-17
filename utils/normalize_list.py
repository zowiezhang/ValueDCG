import numpy as np

def normalize_list(input_list):
    # Convert input list to numpy array
    input_array = np.array(input_list)
    
    # NumPy's min and max functions can operate directly on arrays
    min_val = input_array.min()
    max_val = input_array.max()
    
    # Normalization using numpy's broadcasting feature
    # (input_array - min_val) / (max_val - min_val)
    normalized_array = (input_array - min_val) / (max_val - min_val)
    
    return normalized_array.tolist()  

def standardize_list(data_list):
    data_array = np.array(data_list)

    mean = np.mean(data_array)
    std = np.std(data_array)

    standardized_data = (data_array - mean) / std

    min_abs_value = abs(np.min(standardized_data))
    standardized_data += min_abs_value
    
    return standardized_data.tolist()

