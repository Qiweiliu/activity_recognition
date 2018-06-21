def estimate_kalman_gain(estimate_error, measurement_error):
    return estimate_error / (estimate_error + measurement_error)


def calculate_current_estimate(previous_estimate, kg, measurement):
    return previous_estimate + kg * (measurement - previous_estimate)


def calculate_estimate_error(kg, previous_estimate_error):
    return (1 - kg) * previous_estimate_error


def kalman_filter(measurements, initial_estimate, initial_estimate_error, initial_measurement_error):
    filtered_result = []
    current_estimate = None
    current_estimate_error = None
    current_kg = None
    for i in range(0, len(measurements)):
        if i is 0:
            current_kg = estimate_kalman_gain(initial_estimate_error, initial_measurement_error)
            current_estimate = calculate_current_estimate(initial_estimate, current_kg, measurements[i])
            current_estimate_error = calculate_estimate_error(current_kg, initial_estimate_error)
            print('Current KG: ', current_kg, '\n',
                  'Current estimate: ', current_estimate, '\n',
                  'Current estimate error: ', current_estimate_error
                  )
        else:
            current_kg = estimate_kalman_gain(current_estimate_error, initial_measurement_error)
            current_estimate = calculate_current_estimate(current_estimate, current_kg, measurements[i])
            current_estimate_error = calculate_estimate_error(current_kg, current_estimate_error)
            print('Current KG: ', current_kg, '\n',
                  'Current estimate: ', current_estimate, '\n',
                  'Current estimate error: ', current_estimate_error
                  )
            filtered_result.append(current_estimate)
    return filtered_result


if __name__ == '__main__':
    # measurements = [75, 71, 70, 74]
    #
    # initial_estimate = 68
    # initial_estimate_error = 2
    # initial_measurement_error = 4
    # kalman_filter(measurements,initial_estimate,initial_estimate_error,initial_measurement_error)
    from pykalman import KalmanFilter
    measurements = [[1, 0], [0, 0], [0, 1]]
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=2)
    kf.em(measurements).smooth([[2, 0], [2, 1], [2, 2]])[0]
