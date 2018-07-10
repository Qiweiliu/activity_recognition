import unittest
import src.signal_process_tools as spt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter


class TestSignalProcessTools(unittest.TestCase):

    def setUp(self):
        self.data = np.array(np.loadtxt('../dataFeb/AmplitudeNov19_2stand.out', dtype=float, delimiter=','))
        # self.data = spt.remove_background(self.data)
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
        tmp = spt.compute_amplitude_difference_range(np.absolute(self.data))
        distribution = np.histogram(tmp)
        # plt.hist(x=tmp, bins=distribution[1], density=True)
        plt.scatter(np.arange(0, tmp.size), tmp)
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

    def test_extract_features(self):
        data = spt.process_signals(self.data)
        x, y = spt.interpolation_one_d(data[1])
        result = spt.compute_velocity(x, y)
        resultt = spt.compute_acceleration(x, y)

        # plt.plot(x, y, label='P')

        # distribution = np.histogram(result[1])
        # plt.hist(x=result[1], bins=distribution[1], density=True)
        # print(np.percentile(98,))
        # plt.scatter(np.arange(0, result[1].size), result[1])
        # plt.show()

        plt.plot(result[0], result[1], label='V')
        plt.plot(resultt[0], resultt[1], label='A')
        plt.legend()
        plt.show()

    def test_freq_feature(self):
        self.data = np.array(np.loadtxt('../dataFeb/AmplitudeNov19_2stand.out', dtype=float, delimiter=','))
        self.dataa = np.array(np.loadtxt('../dataFeb/AmplitudeNov19_2Pickup.out', dtype=float, delimiter=','))
        data = spt.process_signals(self.data)
        dataa = spt.process_signals(self.dataa)
        x, y = spt.interpolation_one_d(data[1])
        xx, yy = spt.interpolation_one_d(dataa[1])

        plt.subplot(211)
        plt.plot(np.fft.fftfreq(y.size, 1 / 16), np.absolute(np.fft.fft(y)))
        plt.subplot(212)
        plt.plot(np.fft.fftfreq(yy.size, 1 / 16), np.absolute(np.fft.fft(yy)))
        plt.show()


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.data = np.absolute(
            np.array(
                np.loadtxt('../dataFeb/AmplitudeFeb9_1walk.out', dtype=float, delimiter=',')
            )
        )
        self.data = np.absolute(spt.remove_background(self.data))

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

    def test_observe_amplitudes_changes(self):
        tmp = spt.compute_amplitude_difference_range(self.data)

        motion_indices = self.extract_motion_part(tmp, 0.8).tolist()
        c_list = [2] * len(tmp)
        for i in motion_indices:
            c_list[i] = 1

        plt.scatter(np.arange(0, tmp.size), tmp, c=c_list)
        plt.show()

    def extract_motion_part(self, data, std_times):
        std_am = np.std(data)
        med_am = np.median(data)
        motion_am = np.where(
            (data > (med_am + std_times * std_am)))
        return motion_am[0]

    def test_plot_surface(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = np.sin(R)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
        pass

    def test_learn_meshgird(self):
        x = np.arange(0, 1000)
        y = np.arange(0, 1000)
        xx, yy = np.meshgrid(x, y)
        plt.scatter(xx, yy)
        plt.show()


if __name__ == '__main__':
    unittest.main()
