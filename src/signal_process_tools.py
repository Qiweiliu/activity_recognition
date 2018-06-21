import numpy as np
import matplotlib.pyplot as plt

def remove_background(data):
    diff = np.diff(data, axis=0)
    # max_diff = np.argmax(diff, axis=1)
    return diff


def slice_data(data, x, y):
    """
    Slice data in order to the length of row and column are divisible by x and y
    :param data:
    :param x: average every x elements across columns
    :param y: average every y elements across rows
    :return: sliced data
    """
    x_curtailment = data.shape[1] % x
    y_curtailment = data.shape[0] % y
    data = data[0:data.shape[0] - y_curtailment, 0:data.shape[1] - x_curtailment]
    return data


def attenuate_noise(data, x, y):
    def average_across_column(row, size):
        return np.mean(row.reshape(row.size // size, size), axis=1)

    data = slice_data(data, x, y)

    # average along x-axis of data
    averaged_diff = np.apply_along_axis(average_across_column, 1, data, x)

    # average along y-axis of data
    averaged_diff = np.apply_along_axis(average_across_column, 0, averaged_diff, y)

    return averaged_diff


def take_index(row, threshold):
    """
    Take indices from candidate indices array.
    the n represent the number of largest amplitudes that calculated
    by the threshold. Then taking the indices of the n-largest amplitudes.

    :param row:
    :param threshold:
    :return:
    """
    indices = np.argsort(row)
    n = int(row.size * (1 - threshold))
    return np.amin(indices[-n:])


def taking_motion_index(take_index, data, threshold):
    """
    The function takes indices of max amplitudes for each PRI.
    Note that, the each tuples contains of the data contain 8192 or 4096 samples

    :param take_index:
    :param data:
    :param threshold:
    :return: the return is a 1-d array that consists of the indices of max amplitudes in each tuple
    """

    return np.apply_along_axis(func1d=take_index,
                               axis=1,
                               arr=data,
                               threshold=threshold
                               )


def correct_outliers(indices, window_length, threshold):
    def remove_outliers(segment, threshold):
        median_loc = np.median(segment)
        outlier_indices = np.where(
            (segment > (median_loc + threshold))
            |
            (segment < (median_loc - threshold)))

        return np.delete(segment, outlier_indices)

    new_indices = []
    i = 0
    for segment in np.array_split(indices, window_length):
        if i > 1:
            new_indices = remove_outliers(new_indices, i * threshold)

        new_indices = np.hstack((new_indices, remove_outliers(segment, threshold)))
        i += 1

    return new_indices


def compute_amplitude_range(data):
    def subtract(row):
        # row = np.absolute(row)
        indices = np.argsort(row)
        plt.plot(row)
        plt.annotate("max",xy=(indices[-1],row[indices[-1]]))
        plt.annotate("min", xy=(indices[0], row[indices[0]]),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.show()
        return np.absolute(indices[0]-indices[-1])

    return np.apply_along_axis(func1d=subtract,
                               axis=1,
                               arr=data
                               )

