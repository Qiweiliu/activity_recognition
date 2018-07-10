import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def interpolation_one_d(motion_indices):
    x = np.arange(0, motion_indices.size)
    f = interp1d(x, motion_indices, kind='cubic')
    x_new = np.arange(0, motion_indices.size - 1, 0.1)
    y_new = f(x_new)

    return x_new, y_new


def compute_velocity(x, y):
    x = x[0:-1] + np.diff(x) / 2
    y = set_savgol(np.diff(y) / x[1], 4)
    return x, y


def compute_acceleration(x, y):
    x, y = compute_velocity(x, y)
    x = x[0:-1] + np.diff(x) / 2
    y = set_savgol(np.diff(y) / x[1], 4)
    return x, y


def extract_features():
    pass


class BackgroundRemover:
    def remove(self, data):
        """
        The function remove the static background reflection from the measurements.
        Note that the all the values are absolute because the output format of the Walabot
        :param data:
        :return:
        """

        # Taking absolute value of the data before processing
        data = np.absolute(data)

        # After removing the background, still go head to take absolute value
        data = np.absolute(remove_background(data))
        return data


class NoiseAttenuator:
    def __init__(self):
        pass

    def _slice_data(self, data, fast_time, slow_time):
        """
        Slice data in order to the length of row and column are divisible by x and y
        :param data:
        :param fast_time: average every x elements across columns
        :param slow_time: average every y elements across rows
        :return: sliced data
        """
        x_curtailment = data.shape[1] % fast_time
        y_curtailment = data.shape[0] % slow_time
        data = data[0:data.shape[0] - y_curtailment, 0:data.shape[1] - x_curtailment]
        return data

    def attenuate(self, data, fast_time, slow_time):
        # Attenuate noises by average per n adjacent times scan
        def average_across_column(row, size):
            return np.mean(row.reshape(row.size // size, size), axis=1)

        data = self._slice_data(data, fast_time, slow_time)

        # average along x-axis of data
        averaged_diff = np.apply_along_axis(average_across_column, 1, data, fast_time)

        # average along y-axis of data
        averaged_diff = np.apply_along_axis(average_across_column, 0, averaged_diff, slow_time)

        return averaged_diff


class SignalSmoother:
    """
    This smoother smooth signals using savgol algorithm
    """

    def __init__(self, p):
        self.p = p

    def smooth(self, motion_path, static_indices_range):
        """
        Only smooth the dynamic section of a whole motion path
        :param motion_path: The coarse  motion path before smoothing
        :param static_indices_range: The array of tuples contains the motion ranges
        :return: Motion_path include the complete motion path. The smoothed path only return
        the section of path that is smoothed
        """
        start = 0
        end = 0

        head_tuple = static_indices_range[0]
        tail_tuple = static_indices_range[-1]

        if head_tuple[0] == 0 and tail_tuple[1] == motion_path.size - 1:
            start = static_indices_range[0][1] + 1
            end = static_indices_range[-1][0] + 1

        if head_tuple[0] == 0 and tail_tuple[1] != motion_path.size - 1:
            start = static_indices_range[0][1] + 1
            end = static_indices_range[-1][1] + 1

        if head_tuple[0] != 0 and tail_tuple[1] == motion_path.size - 1:
            start = 0
            end = tail_tuple[0] + 1

        if head_tuple[0] != 0 and tail_tuple[1] != motion_path.size - 1:
            start = 0
            end = tail_tuple[1]

        smooth_target = motion_path[start:end]

        smoothed_path = self._set_savgol(smooth_target, self.p)

        np.put(motion_path, np.arange(start, end), smoothed_path)

        return motion_path, smoothed_path

    def _set_savgol(self, data, p):
        if data.size % 2 == 0:
            smoothed = savgol_filter(data, data.size - 1, p)
        else:
            smoothed = savgol_filter(data, data.size, p)
        return smoothed


class MotionAnalyzer:
    def __init__(self, smoother, fast_time_threshold, motion_amplitude_threshold, only_return_motions=True):
        self.only_return_motions = only_return_motions
        self.motion_amplitude_threshold = motion_amplitude_threshold
        self.fast_time_threshold = fast_time_threshold
        self.smoother = smoother

    def extract(self, data):
        """"""
        '''
        Preliminarily discovering all possible motion index comparing each scan based
        on the possible range of indices because of one's volume
        '''
        possible_indices = taking_motion_index(self._take_index, data, threshold=self.fast_time_threshold)

        '''
        Finding the motion part of the time-series data. 
        This steps return a list of amplitudes of the change betweentwo scans
        '''
        amplitude_range = self._compute_amplitude_difference_range(data)

        # Eliminate the indices that are thought as no motions
        motion_indices = self._extract_motion_indices(amplitude_range, self.motion_amplitude_threshold).tolist()

        # Set static value of static positions as zero
        motion_path = self._set_static_indices(possible_indices, motion_indices)

        # Interpolating static history
        motion_path, static_indices_range = self._interpolate_static_history(motion_path)

        # Smooth the curve
        motion_path, smoothed_path = self.smoother.smooth(motion_path, static_indices_range)

        if self.only_return_motions:
            return smoothed_path
        else:
            return motion_path

    def _interpolate_static_history(self, motion_indices):
        """
        This function interpolates the positions that are not shown due to static
        The length of the motion_indices shall be at least 3
        :param motion_indices:
        :return:
        """
        static_indices = discover_static_status(motion_indices)

        for start, end in static_indices:
            if start == 0 and (end + 1) < motion_indices.size:
                interpolated_position = motion_indices[end + 1]
                for i in np.arange(start, end + 1):
                    motion_indices[i] = interpolated_position

            if end == motion_indices.size - 1:
                interpolated_position = motion_indices[start - 1]
                for i in np.arange(start, end + 1):
                    motion_indices[i] = interpolated_position

            if start != 0 and end != motion_indices.size - 1:
                interpolated_position = (motion_indices[start - 1] + motion_indices[end + 1]) / 2
                for i in np.arange(start, end + 1):
                    motion_indices[i] = interpolated_position

        return motion_indices, static_indices

    def _extract_motion_indices(self, data, std_times):
        """
        This function return indices that meet
        the criterion that a index can be thought
        as a "motion"
        :param data:
        :param std_times:
        :return:
        """
        std_am = np.std(data)
        med_am = np.median(data)
        motion_am = np.where(
            (data > (med_am + std_times * std_am)))
        return motion_am[0]

    def _set_static_indices(self, possible_indices, motion_indices):
        for i in np.arange(0, possible_indices.size):
            if i not in motion_indices:
                possible_indices[i] = 0

        return possible_indices

    def _take_index(self, row, threshold):
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

    def _taking_motion_index(self, take_index, data, threshold):
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

    def _compute_amplitude_difference_range(self, data):
        """
        Return max amplitude of the current fast-time signals
        :param data:
        :return:
        """

        def subtract(row):
            return np.max(row)

        return np.apply_along_axis(func1d=subtract,
                                   axis=1,
                                   arr=data
                                   )


class DataProcessor:
    def __init__(self, background_remover, motion_analyzer, ):
        self.motion_analyzer = motion_analyzer
        self.background_remover = background_remover

    def process(self, data_set):
        """

        Current Hypotheses:
            1. The motion of the object is assumed as continuous without interrupting in the middle
            2. The motion signals take up small portion of the whole slow-time signals
        :param data_set:
        :return:
        """
        background_removed_data = self.background_remover.remove(data_set)
        motions = self.motion_analyzer.extract(background_removed_data)
        return motions
