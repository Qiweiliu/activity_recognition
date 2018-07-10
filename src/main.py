import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import src.signal_process_tools as spt
from scipy.interpolate import interp1d

sns.set()

data = np.array(np.loadtxt('../dataFeb/AmplitudeNov19_2stand.out', dtype=float, delimiter=','))

# data = np.array(np.load('../dataFeb/2IPhone_breath_0.npy').item()['walabot'])


if __name__ == '__main__':
    background_remover = spt.BackgroundRemover()
    smoother = spt.SignalSmoother(p=4)
    motion_analyzer = spt.MotionAnalyzer(
        smoother=smoother,
        fast_time_threshold=0.99,
        motion_amplitude_threshold=0.8,
        only_return_motions=True
    )
    data_processor = spt.DataProcessor(
        background_remover=background_remover,
        motion_analyzer=motion_analyzer,
    )

    motion_path = data_processor.process(data)

    # sns.heatmap(data_a[0].transpose()[0:1000], cmap=sns.color_palette("cubehelix", 128))

    # x, y = spt.interpolation_one_d(motion_path)
    # plt.plot(x, y)
    plt.plot(motion_path)
    plt.show()
