import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import linalg
from math import ceil
from math import factorial
from scipy import signal


# read files
def readfile(file):
    sd = []
    with open(file, 'r') as f:
        for x in f:
            sd.append([float(n) for n in x.strip().split(',')])
    # sd = np.reshape(sd, -1)
    return sd


# calculate the distance between sensor and target
def GetDistance(listD, listT, index):
    t0 = 9.765625000e-12
    c = 299792458
    ux = listD[index].index(np.max(listD[index]))
    distance = (listT[index][ux] - t0) * c
    return distance


# draw a picture
def drawpicture(list):
    y = np.linspace(0, len(list), len(list))
    fig = plt.figure()
    ax1 = plt.subplot()
    # ax1.set_ylim([-2, 2])
    ax1.plot(y, list)
    plt.show()


# butterworth filter  cutt_freq = 0.33
def frequencyoutput(y):
    yx = np.array(y) / 350
    t = np.arange(len(yx))
    fs = len(yx)
    Ts = 1 / fs
    cuttoff_freq = 0.33
    samp_rate = 20
    norm_pass = cuttoff_freq / (samp_rate / 2)
    norm_stop = 1.5 * norm_pass
    (N, Wn) = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30, analog=0)
    (b, a) = signal.butter(N, Wn, btype='low', analog=0, output='ba')
    yaxis = signal.lfilter(b, a, yx)
    yfreq = np.fft.fft(yaxis)
    return yfreq, y, yaxis


# FFT transform, return to
def getfftradio(signaldata):
    Tts = 1 / len(signaldata)
    yft = np.fft.fft(signaldata)
    xft = np.fft.fftfreq(len(yft), Tts)
    fftmax = np.max(yft)
    fftmin = np.min(yft)
    fftratio = np.max(np.abs(fftmax / fftmin))
    return fftratio, fftmax, fftmin, yft


# collect the main features for motion
def ouput_bootsing_feature(yi):
    t = remove_outrange(yi[0])
    disy = np.array(t) / 350
    maxamplitude = np.max(yi[1])
    fftradio = getfftradio(t)
    filted = filterbasesavutzky(disy)
    velocity = getVelocityforCART(filted)
    maxdisplacement = np.max(t)
    time_span = max_time_span(t)
    slope = slopefeature(t)
    return [time_span, maxamplitude, fftradio[1], fftradio[2], fftradio[0], maxdisplacement, velocity, slope[1]]


# filter function remove the out of range data
# example: [300,0,0,435] the two-middle [0] will be considered as outliers data
def remove_outrange(yi0):
    op = []
    op.append(yi0[0])
    for x in range(1, len(yi0) - 1):
        if np.abs(yi0[x] - yi0[x - 1]) > 500 and np.abs(yi0[x + 1] - yi0[x]) > 500:
            if yi0[x - 1] == 0 or yi0[x + 1] == 0:
                op.append(yi0[x])
            else:
                u = int((yi0[x - 1] + yi0[x + 1]) / 2)
                op.append(u)
        elif yi0[x] == 0:
            if yi0[x - 1] > 0.01 and yi0[x + 1] > 0.01:
                u = int((yi0[x - 1] + yi0[x + 1]) / 2)
                op.append(u)
            else:
                op.append(yi0[x])
        else:
            op.append(yi0[x])
    b = gettheconsecutivelist(op)
    for x in range(1, len(b)):
        if b[x][0] - b[x - 1][-1] <= 3:
            op[b[x][0] - 1] = int((op[b[x][0]] + op[b[x - 1][-1]]) / 2)
            op[b[x][0] - 2] = op[b[x][0] - 1]
    return op


# calcualte the slope of motion images
def slopefeature(t):
    b = gettheconsecutivelist(t)
    maxlist = max(b, key=len)
    outputlist = []
    for x in maxlist:
        outputlist.append(t[x])
    slopediff = np.diff(outputlist)
    slope = np.sum(slopediff) / len(slopediff)
    return outputlist, slope, slopediff, np.sum(slopediff), b


# get K-means classfication points
def getClassificationpoint(ulabel):
    gh = []
    for x in range(len(ulabel) - 2):
        if ulabel[x] - ulabel[x + 1] != 0 and ulabel[x + 1] == ulabel[x + 2]:
            gh.append(x)
    return gh


# interval time between two-neighbor pulse
def interval_time(totaltime, list, fasttime):
    yui = (totaltime - fasttime * len(list)) / (len(list) - 1)
    return yui


# get the divergent points between two-neighbor pulses
def alldifferentpoint(list):
    difference = []
    maxdifferenceloc = []
    maxdifference = []
    op = 0
    for y in range(len(list) - 1):
        for x in range(len(list[1])):
            difference.append(float("%.2f" % (list[y][x] - list[y + 1][x])))
        if np.max(difference) == 0:
            maxdifferenceloc.append(0)
            op += 1
        else:
            maxdifferencev = difference.index(np.max(difference))
            maxdifferenceloc.append(maxdifferencev)
        maxdifference.append(np.max(difference))
        difference = []
    return maxdifferenceloc, maxdifference, op


def getintervaldifferent(list):
    u = []
    for x in range(len(list) - 1):
        u.append((list[x + 1] - list[x]))
    return u


# reformat distence to meters
def convertmeters(yi, timedata):
    disy = []
    t0 = 9.765625000e-12
    c = 299792458
    for x in range(len(yi)):
        disy.append((timedata[x][yi[x]] - t0) * c)
    return disy


# get k-means cluster center points
def getcluster_center(cluster, disy):
    disy = np.array(disy)
    kmeans = KMeans(n_clusters=cluster, random_state=None)
    u = kmeans.fit(disy.reshape(-1, 1))
    cluster_center = u.cluster_centers_
    ulabel = u.labels_
    return cluster_center, ulabel


def filter(df):
    cv = []
    for x in range(len(df) - 1):
        if df[x + 1] - df[x] >= 10:
            cv.append(df[x])
    return cv


# filter based savutzky algorithm
def filterbasesavutzky(disy):
    ta = np.array(disy)
    disy = savitzky_golay(ta, 35, 3)
    disy = disy.reshape(-1)
    disy = np.array(disy).tolist()
    return disy


# save amplituate data
def addsaveamplituate(list):
    with open('Data/Data.csv', 'a') as f:
        for x in range(len(list)):
            output_string = str('%.18f' % list[x])
            output_string += "\n"
            f.writelines(output_string)
    f.close()


# save velocity data
def addsaveVelocity(list):
    with open('Data/Velocity.csv', 'a') as f:
        for x in range(len(list)):
            output_string = str('%.18f' % list[x])
            output_string += "\n"
            f.writelines(output_string)
    f.close()


def addsaveA(list):
    with open('Data/Amplitude.csv', 'w') as f:
        for x in range(len(list)):
            output_string = str('%.18f' % list[x])
            output_string += "\n"
            f.writelines(output_string)
    f.close()


# save motions' label
def addsavelabel(list):
    with open('Data/label.csv', 'w') as f:
        for x in range(len(list)):
            output_string = str('%.18f' % list[x])
            output_string += "\n"
            f.writelines(output_string)
    f.close()


# get velocity
def getVelocityforCART(output):
    t_end = 7.999023438e-08  # time[1][-1]
    interval_times = interval_time(60, output, t_end)
    velocity = []
    for x in range(len(output) - 1):
        velocity.append((np.abs(output[x + 1] - output[x]) / interval_times))
    return np.max(velocity)


# get the consecutives group number
def group_consecutives(vals, step=1):
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


# Get the max time_span
def max_time_span(yi0):
    vaildable = gettheconsecutivelist(yi0)
    u = []
    for x in range(len(vaildable)):
        u.append(len(vaildable[x]))
    return (np.max(u))


# get all consecutive number list
def gettheconsecutivelist(list):
    indexlist = []
    for x in range(len(list)):
        if list[x] > 0.01:
            indexlist.append(x)
    return group_consecutives(indexlist)


# feature filter
def featurefilter(list):
    featurefilterlist = []
    for x in range(len(list)):
        if len(list[x]) > 10:
            featurefilterlist.append(list[x])
    return featurefilterlist


# verify the sit-stand and stand-to-sit (old version)
def featureresult(filter1, list):
    increase = 0
    decrease = 0
    resultfeature = []
    roui = []
    variableplus = []
    variabledown = []
    for y in range(len(filter1)):
        for x in range(len(filter1[y]) - 1):
            roui.append(list[filter1[y][x]])
            t = list[filter1[y][x + 1]] - list[filter1[y][x]]
            if t > 0:
                increase += 1
                variableplus.append(t)
            else:
                decrease += 1
                variabledown.append(t)
        rangefirst = list[filter1[y][0]]
        rangelast = list[filter1[y][-1]]
        colorfirst = filter1[y][0]
        colorlast = filter1[y][-1]
        mind = np.min(roui)
        maxd = np.max(roui)
        if len(variableplus) > 0:
            meanplus = np.mean(variableplus)
        else:
            meanplus = 0
        if len(variabledown) > 0:
            meandown = np.mean(variabledown)
        else:
            meandown = 0
        # meandown = np.mean(variabledown)
        resultfeature.append([meanplus, meandown, rangefirst, rangelast, mind, maxd, colorfirst, colorlast])
        roui = []
        variableplus = []
        variabledown = []
        increase = 0
        decrease = 0
    return resultfeature


def verifystaticstatus(result2, list):
    increase = 0
    decrease = 0
    increasecolor = []
    decreasecolor = []
    for x in range(len(result2)):
        if result2[x][2] - result2[x][3] > 0.1:
            if result2[x][3] - result2[x][4] > 100:
                middlenumber = int(np.abs(result2[x][7] - result2[x][6]) / 2) + np.min([result2[x][6], result2[x][7]])
                increase += 1
                decrease += 1
                if list[middlenumber] < list[result2[x][6]]:
                    decreasecolor.append([result2[x][6], middlenumber])
                    increasecolor.append([middlenumber, result2[x][7]])
                else:
                    increasecolor.append([result2[x][6], middlenumber])
                    decreasecolor.append([middlenumber, result2[x][7]])

            else:
                decrease += 1
                decreasecolor.append([result2[x][6], result2[x][7]])
        elif result2[x][3] - result2[x][2] > 0.1:
            if result2[x][2] - result2[x][4] > 100:
                middlenumber = int(np.abs(result2[x][7] - result2[x][6]) / 2) + np.min([result2[x][6], result2[x][7]])
                increase += 1
                decrease += 1
                if list[middlenumber] > list[result2[x][6]]:
                    increasecolor.append([result2[x][6], middlenumber])
                    decreasecolor.append([middlenumber, result2[x][7]])
                else:
                    decreasecolor.append([result2[x][6], middlenumber])
                    increasecolor.append([middlenumber, result2[x][7]])


            else:
                increase += 1
                increasecolor.append([result2[x][6], result2[x][7]])
        else:
            increase += 0
            decrease += 0
    return increase, decrease, increasecolor, decreasecolor


# mark the crest and peak of reciprocate motion
def colorpoint(status):
    increaselist = []
    a = []
    b = []
    decreaselist = []
    for x in range(len(status[2])):
        for y in range(status[2][x][0], status[2][x][1] + 1):
            a.append(y)
        increaselist.append(a)
        a = []
    for x in range(len(status[3])):
        for y in range(status[3][x][0], status[3][x][1] + 1):
            b.append(y)
        decreaselist.append(b)
        b = []
    return increaselist, decreaselist


def peakdetectioncrestfrist(cv, disy):
    i = []
    u = []
    # if len(cv) % 2 == 0:
    i.append([np.max(disy[:cv[0]]), disy.index(np.max(disy[:cv[0]]))])
    for x in range(1, len(cv)):
        if x % 2 == 1:
            maxd = np.max(disy[cv[x - 1]:cv[x]])
            # print(cv[x - 1], cv[x])
            i.append([maxd, cv[x - 1] + disy[cv[x - 1]:cv[x]].index(maxd)])
        else:
            mind = np.min(disy[cv[x - 1]:cv[x]])
            # print(cv[x - 1], cv[x])
            u.append([mind, cv[x - 1] + disy[cv[x - 1]:cv[x]].index(mind)])
    return i, u


def peakdetectiontroughfirst(cv, disy):
    i = []
    u = []
    # if len(cv) % 2 == 0:
    i.append([np.max(disy[:cv[0]]), disy.index(np.max(disy[:cv[0]]))])
    for x in range(1, len(cv)):
        if x % 2 == 0:
            maxd = np.max(disy[cv[x - 1]:cv[x]])
            # print(cv[x - 1], cv[x])
            i.append([maxd, cv[x - 1] + disy[cv[x - 1]:cv[x]].index(maxd)])
        else:
            mind = np.min(disy[cv[x - 1]:cv[x]])
            # print(cv[x - 1], cv[x])
            u.append([mind, cv[x - 1] + disy[cv[x - 1]:cv[x]].index(mind)])
    return i, u


def getxoutput(output, disy):
    df = getClassificationpoint(output[1])
    cv = []
    for x in range(len(df) - 1):
        if df[x + 1] - df[x] >= 10:
            cv.append(df[x])
    cv.append(len(output[1]) - 1)
    if cv[0] < 5:
        cv.remove(cv[0])
        print('Removed first one from fc')
    if output[0][0] > output[0][1] and output[1][cv[0] - 1] == 1:
        xoutput = peakdetectioncrestfrist(cv, disy)
    elif output[0][0] < output[0][1] and output[1][cv[0] - 1] == 0:
        xoutput = peakdetectioncrestfrist(cv, disy)
    else:
        xoutput = peakdetectiontroughfirst(cv, disy)
    if xoutput[1][0][1] > xoutput[0][1][1]:
        xoutput[0].remove(xoutput[0][0])
        print('Removed a mistake from fc')
    return xoutput


def getVelocity(xoutput, intervaltime):
    mark1 = False
    mark0 = False
    markq = False
    velcocitylisto = []
    if len(xoutput[1]) > len(xoutput[0]):
        a = len(xoutput[0])
        mark0 = True
    elif len(xoutput[1]) == len(xoutput[0]):
        a = len(xoutput[1])
        markq = True
    else:
        a = len(xoutput[1])
        mark1 = True
    for x in range(a):
        distenceF = np.abs(xoutput[0][x][0] - xoutput[1][x][0])
        timeF = (np.abs(xoutput[1][x][1] - xoutput[0][x][1])) * intervaltime
        velcocitylisto.append(distenceF / (timeF / 2))
        if markq == True:
            if x + 1 < a:
                distenceF = np.abs(xoutput[0][x + 1][0] - xoutput[1][x][0])
                timeF = (np.abs(xoutput[1][x][1] - xoutput[0][x + 1][1])) * intervaltime
                velcocitylisto.append(distenceF / (timeF / 2))
        if mark1 == True:
            distenceF = np.abs(xoutput[0][x + 1][0] - xoutput[1][x][0])
            timeF = (np.abs(xoutput[1][x][1] - xoutput[0][x + 1][1])) * intervaltime
            velcocitylisto.append(distenceF / (timeF / 2))
        if mark0 == True:
            distenceF = np.abs(xoutput[0][x][0] - xoutput[1][x + 1][0])
            timeF = (np.abs(xoutput[1][x + 1][1] - xoutput[0][x][1])) * intervaltime
            velcocitylisto.append(distenceF / (timeF / 2))
    # velcocitylisto = []
    # for x in range(len(xoutput[0])):
    #     distenceF = np.abs(xoutput[0][x][0] - xoutput[1][x][0])
    #     timeF = (np.abs(xoutput[1][x][1] - xoutput[0][x][1])) * intervaltime
    #     velcocitylisto.append(distenceF / (timeF / 2))
    return velcocitylisto


def peakdraw(xoutput, disy):
    pointx = []
    pointy = []
    pointx1 = []
    pointy1 = []
    for x in range(len(xoutput[0])):
        pointx.append(xoutput[0][x][1])
        pointy.append(xoutput[0][x][0])
    for x in range(len(xoutput[1])):
        pointx1.append(xoutput[1][x][1])
        pointy1.append(xoutput[1][x][0])
    y = np.linspace(0, len(disy), len(disy))
    fig = plt.figure()
    ax1 = plt.subplot()
    # ax1.set_ylim([-2, 2])
    ax1.plot(y, disy, )
    ax1.scatter(pointx, pointy, color='red')
    ax1.scatter(pointx1, pointy1, color='green')
    plt.show()


# filter for lowess
def lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    # except ValueError, msg:
    #     raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')
