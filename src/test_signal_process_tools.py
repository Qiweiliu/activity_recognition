import unittest
import src.signal_process_tools as spt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation


class TestSignalProcessTools(unittest.TestCase):

    def setUp(self):
        self.data = np.array(np.loadtxt('../dataFeb/AmplitudeNov19_2stand.out', dtype=float, delimiter=','))
        self.data = spt.remove_background(self.data)
        # self.data = spt.attenuate_noise(self.data, 1, 4)

    def test_take_index(self):
        mock_data = np.array([5, 2, 3, 7, 11, 1])
        index = np.argsort(mock_data)
        print(mock_data)
        print(np.sort(mock_data))
        print(index)
        # print(mock_data[index])
        # print(np.amax(index))

    def test_correct_outliers(self):
        data = np.array([0, 335, 272, 1553, 442, 442, 442, 425, 353, 204, 426, 382, 317, 309, 1583, 681, 301])
        print(spt.correct_outliers(data, 3, 300))

    def test_interpolation(self):
        data = np.array([0, 335, 272, 1553, 442, 442, 442, 425, 353, 204, 426, 382, 317, 309, 1583, 681, 301])
        x = np.arange(0, data.size)
        f = interp1d(x, data)
        x_new = np.arange(0, data.size)
        y_new = f(x_new)
        plt.plot(x, data, 'o', x_new, y_new, '-')
        plt.show()

    def test_get_amplitude_range(self):
        tmp = spt.compute_amplitude_range(self.data)
        distribution = np.histogram(tmp)
        # plt.hist(x=tmp, bins=distribution[1], density=True)
        plt.plot(tmp)
        plt.show()
        # print(np.percentile(tmp,80))
        # for i in range(0,self.data.size):
        #
        #     plt.plot(self.data[i])
        #     plt.show()

    def test_split(self):
        data = np.array([0, 335, 272, 1553, 442, 442, 442, 425, 353, 204, 426, 382, 317, 309, 1583, 681, 301])
        sliced = np.array_split(data, 4)
        # print(np.array_split(data, 4))
        save = []
        for element in sliced:
            save = np.hstack((save, element))

        print(save)
        # np.nditer(np.array_split(data, 4))

        # print((map(self.temp, ))


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.data = np.array(np.loadtxt('../dataFeb/AmplitudeNov19_2stand.out', dtype=float, delimiter=','))
        # self.data = spt.remove_background(self.data)

    def test_observe(self):
        def change(count):
            self.signals.set_ydata(np.absolute(self.data[count][200:500]))
            self.signals_b.set_ydata(self.data[count][200:500])

        fig, ax, = plt.subplots(1)
        self.signals, = ax.plot(np.absolute(self.data[0][200:500]))
        self.signals_b, = ax.plot(self.data[0][200:500])
        ax.axhline(y=0)
        steps = np.arange(0, 8191)
        ani = FuncAnimation(fig, change, steps, interval=400, repeat=False)

        plt.show()


if __name__ == '__main__':
    unittest.main()
