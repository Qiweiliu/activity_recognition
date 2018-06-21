import src.functionclass as fc
import glob
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# walkdata_names = sorted(glob.glob('DataJan/Amplitude*sit.out'))

walkdata_names = sorted(glob.glob('../DataFeb/AmplitudeNov19_2stand.out'))

print(walkdata_names)
# read signal raw data
signaldata = fc.readfile(walkdata_names[0])
diff = np.diff(signaldata,axis=0)

# extract the motion part from the raw data (remove the background)
yi = fc.alldifferentpoint(signaldata)
# get the amplitude data of motion part
amplitudet = yi[1]
# get the relative displacement of motion
motion_t = yi[0]
# remove the out range date

t = np.array(fc.remove_outrange(motion_t))
# t = np.array(motion_t)


tmp = fc.max_time_span(t)

def regression(t):
    t = np.mean(t.reshape(t.size // 2, 2), axis=1)
    regr = linear_model.LinearRegression()
    t = np.array(t).reshape(len(t), 1)[0:70]
    # print(t)
    x = np.arange(0, len(t)).reshape(len(t), 1)
    regr.fit(x, t)
    print(regr.coef_)

    plt.plot(x, t)
    plt.plot(x, regr.predict(x), color='blue', linewidth=3)
    plt.show()

print()
