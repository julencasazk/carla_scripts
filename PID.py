
class PID:
    def __init__(self, kp, ki, kd, N, Ts, u_min, u_max, kb_aw, der_on_meas=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.n = N
        self.ts = Ts
        self.min_output = u_min
        self.max_output = u_max
        self.kb_aw = kb_aw
        self.der_on_meas = der_on_meas

        self.i_state = 0.0
        self.e_prev = 0.0
        self.d_prev = 0.0
        self.w_prev = 0.0
    

        self._update_derivative_coeffs()

    def _update_derivative_coeffs(self):
        Kd = self.kd
        N = self.n
        Ts = self.ts
        if Kd <= 0.0 or N <= 0.0:
            self.k_u = 0.0
            self.k_w = 0.0
        else:
            self.k_u = -((N * Ts - 2.0) / (N * Ts + 2.0))     # Tustin
            self.k_w = (2.0 * Kd) / (N * Ts + 2.0)

    def step(self, setpoint, measurement):
        e = setpoint - measurement

        u_p = self.kp * e

        
        w = e if self.der_on_meas else (-measurement) # derivative on error
        u_d = self.k_u * self.d_prev + self.k_w * w - self.k_w * self.w_prev
        self.d_prev = u_d
        self.w_prev = w

        dI_base = self.ki * self.ts * e  # backward Euler
        u_i = self.i_state + dI_base

        u_unsat = u_p + u_i + u_d
        u = max(self.min_output, min(self.max_output, u_unsat))

        aw = self.kb_aw * (u - u_unsat)  # back-calculation
        self.i_state = u_i + aw

        self.e_prev = e
        return u
