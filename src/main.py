import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import src.signal_process_tools as spt
from scipy.interpolate import interp1d

sns.set()

data = np.array(np.loadtxt('../dataFeb/AmplitudeNov19_0Falling.out', dtype=float, delimiter=','))


def process_signals(data):
    data = spt.remove_background(data)

    data = spt.attenuate_noise(data, 1, 4)

    max_indices = spt.taking_motion_index(spt.take_index, data, threshold=0.99)

    corrected_indices = spt.correct_outliers(max_indices, max_indices.size // 16, 300)

    # interpolation
    x = np.arange(0, corrected_indices.size)
    f = interp1d(x, corrected_indices)
    x_new = np.arange(0, corrected_indices.size - 1, 0.1)
    interpolated_indices = f(x_new)

    smooth_length = corrected_indices.size
    if corrected_indices.size % 2 is 0:
        smooth_length -= 1
    smoothed_indices = savgol_filter(corrected_indices, smooth_length, 7)
    return data, max_indices, corrected_indices, interpolated_indices, smoothed_indices


data_a = process_signals(data)
data_b = process_signals(np.absolute(data))

# plt.subplot(211)
# sns.heatmap(data_a[0].transpose(), cmap=sns.color_palette("cubehelix", 128))
# plt.plot(data_a[3], label='With N')
#
# plt.subplot(212)
# sns.heatmap(data_b[0].transpose(), cmap=sns.color_palette("cubehelix", 128))
# plt.plot(data_b[3], label='With N')

# plt.plot(data_b[1], label="Without N")
# plt.legend()

# -----------------------------------------
plt.plot(data_b[1])
plt.plot(data_b[2])
plt.plot(data_b[4])

# -----------------------------------------
plt.show()
# ax =
# plt.subplot(311)
# plt.plot(max_indices, label='original')
# plt.subplot(312)
# plt.plot(max_indices[0:16])
# plt.subplot(313)
# plt.plot(max_indices[16:32])
# plt.plot(x_new, interpolated_indices, '-', label='outliers rejected')
# plt.plot(spt.correct_outliers(max_indices, max_indices.size // 16, 150, ), 'o')
# plt.plot(savgol_filter(corrected_indices, smooth_length, 7), 'blue', label='smoothed')
