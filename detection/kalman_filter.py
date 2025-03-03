# kalman_filter.py

class KalmanFilter1D:
    def __init__(self, initial_value=0.0, process_variance=1e-4, measurement_variance=1.0):
        self.x = initial_value
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance

    def update(self, measurement):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P = (1 - K) * self.P
        return self.x
