import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import iqr, median_abs_deviation


vec_resistors = np.array([100.0, 101.0, 100.2, 101.3, 101.7, 100.1, 102.7, 99.9, 100.4, 100.3, 102.9,100.2, 101.4, 101.5, 100.6, 101.5])

def transform_to_resistances(array_of_voltages):
    resh_vec = array_of_voltages.reshape(len(array_of_voltages), 6, 16)
    vec_res_repeated = vec_resistors.repeat(len(array_of_voltages)).reshape(len(array_of_voltages), 16)

    sums = resh_vec.sum(axis=1)
    U = np.ones(shape=sums.shape) * 1024

    repeated_rows = np.repeat((U - sums) * vec_res_repeated, 6, axis=0)
    repeated_rows = repeated_rows.reshape(len(array_of_voltages), 6, 16) + 0.000001


    resistances = (resh_vec / repeated_rows).reshape(len(array_of_voltages), -1)


    return resistances

def compute_power(array):

    data = np.mean(array, axis=0)
    excluded_positions = [15, 31, 47, 63, 93]
    #excluded_positions_flatten = [idx + 96 * i for idx in excluded_positions for i in range(5)]
    included_indices = [i for i in range(96) if i not in excluded_positions]

    # Compute vector lengths along second dimension
    vector_length = np.linalg.norm(data[included_indices])



    return vector_length

def extract_features(X):
    """X shape: (20, 96) -> returns feature vector"""
    features = []
    # 1. Статистические
    stats = np.vstack([
        X.mean(axis=0), 
        #X.std(axis=0), 
        #np.median(X, axis=0),
        X.max(axis=0), 
        #X.min(axis=0), 
        #np.ptp(X, axis=0),
        #iqr(X, axis=0), 
        #median_abs_deviation(X, axis=0)
    ]).flatten()
    
    # 2. Динамические
    #diffs = np.diff(X, axis=0)
    #dynamics = np.vstack([
    #    diffs.mean(axis=0), diffs.std(axis=0), np.max(np.abs(diffs), axis=0)
    #]).flatten()
    
    # 3. Частотные (первые 3 гармоники)
    #fft = np.abs(np.fft.fft(X, axis=0))[1:4].flatten()
    
    return np.concatenate([stats])

def robust_smoothing(data, window_size=5, alpha=0.7):
    """
    Сначала применяет медианный фильтр для удаления выбросов,
    затем экспоненциальное сглаживание
    """
    # Медианная фильтрация (удаляет выбросы)
    med_filtered = median_filter(data, size=(window_size, 1))

    # Экспоненциальное сглаживание
    smoothed = np.zeros_like(med_filtered)
    smoothed[0, :] = med_filtered[0, :]
    for t in range(1, len(med_filtered)):
        smoothed[t, :] = alpha * med_filtered[t, :] + (1-alpha) * smoothed[t-1, :]

    return smoothed

import numpy as np

def standardize_data(data, len_of_trials=1, means=None, stds=None, ignore_zeros=True):
    """
    Стандартизирует данные (z-score) с возможностью использования предварительно вычисленных статистик.
    
    Параметры:
    - data: np.ndarray, форма (N, M) - входные данные
    - len_of_trials: int - длина временного ряда (по умолчанию 1)
    - means: np.ndarray, форма (M,) - предварительно вычисленные средние (если None, вычисляются)
    - stds: np.ndarray, форма (M,) - предварительно вычисленные стандартные отклонения
    - ignore_zeros: bool - если True, нули игнорируются при вычислении статистик, но остаются нулями в результате
    
    Возвращает:
    - standardized_data: стандартизированные данные
    - means: вычисленные/использованные средние
    - stds: вычисленные/использованные стандартные отклонения
    """
    data = data.reshape(-1, 96)  # Приводим к (N, 96)
    standardized_data = np.zeros_like(data, dtype=float)
    
    # Вычисляем статистики, если не предоставлены
    if means is None or stds is None:
        means = np.zeros(data.shape[1])
        stds = np.zeros(data.shape[1])
        
        for i in range(data.shape[1]):
            col = data[:, i]
        
            means[i] = np.mean(col)
            stds[i] = np.std(col)
                
            if stds[i] < 1e-6:
                stds[i] = 1
    
    # Применяем стандартизацию
    for i in range(data.shape[1]):
        col = data[:, i]
        std = stds[i] if stds[i] > 1e-6 else 1  # Защита от деления на 0
        
        standardized_data[:, i] = (col - means[i]) / std
    
    # Возвращаем в исходную форму (если len_of_trials != 1)
    
    return standardized_data, means, stds