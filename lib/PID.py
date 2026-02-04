"""
Discrete PID controller mirroring the project's `pid_stm32` firmware logic,
implemented as a Python class for simulation and tuning.

Usage (minimal)
  pid = PID(kp, ki, kd, N, Ts, u_min, u_max, kb_aw,
            der_on_meas=True, prop_on_meas=False)
  u = pid.step(setpoint, measurement)
"""


class PID:
    def __init__(self, kp, ki, kd, N, Ts, u_min, u_max, kb_aw, der_on_meas=False, prop_on_meas=False, derivative_disc_method="tustin", integral_disc_method="backeuler", anti_windup="backcalc"):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.n = N
        self.ts = Ts
        self.min_output = u_min
        self.max_output = u_max
        self.kb_aw = kb_aw
        self.der_on_meas = der_on_meas
        self.prop_on_meas = prop_on_meas
        self.derivative_disc_method = derivative_disc_method
        self.integral_disc_method = integral_disc_method
        self.anti_windup = anti_windup

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
            if self.derivative_disc_method == "tustin":
                self.k_u = -((N * Ts - 2.0) / (N * Ts + 2.0))     # Tustin
                self.k_w = (2.0 * N * Kd) / (N * Ts + 2.0)
            elif self.derivative_disc_method == "backeuler":
                self.k_u = 1.0 / (1.0 + (N * Ts))                 # backwards euler
                self.k_w = ((Kd * N)/(1.0+(N * Ts)))
            else:
                self.k_u = -((N * Ts) - 1)
                self.k_w = Kd * N

    def set_gains(self, kp: float, ki: float, kd: float, N: float) -> None:
        """Update controller gains (keeps internal states)."""
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.n = float(N)
        self._update_derivative_coeffs()

    def set_gains_bumpless(
        self,
        kp: float,
        ki: float,
        kd: float,
        N: float,
        setpoint: float,
        measurement: float,
        u_prev: float,
    ) -> None:
        """
        Update gains while trying to keep output continuous ("bumpless transfer")
        by adjusting only the integrator state.

        Intended for gain scheduling when switching between tuned PID gains.
        Preserves derivative filter state (d_prev/w_prev), adjusts i_state so that:
            u_unsat_next ~= u_prev
        given the *current* (setpoint, measurement).
        """
        self.set_gains(kp, ki, kd, N)

        e = float(setpoint) - float(measurement)
        w_d = (-1.0 * float(measurement)) if self.der_on_meas else e
        w_p = (-1.0 * float(measurement)) if self.prop_on_meas else e

        u_p = float(self.kp) * float(w_p)

        # Predict derivative term that will be computed on the next step() call
        u_d = float(self.k_u) * float(self.d_prev) + float(self.k_w) * float(w_d) - float(self.k_w) * float(self.w_prev)

        if self.integral_disc_method == "tustin":
            dI_base = 0.5 * float(self.ki) * float(self.ts) * (e + float(self.e_prev))
        elif self.integral_disc_method == "backeuler":
            dI_base = float(self.ki) * float(self.ts) * e
        else:
            dI_base = float(self.ki) * float(self.ts) * float(self.e_prev)

        # step() will compute u_i = i_state + dI_base, so set i_state for continuity.
        self.i_state = float(u_prev) - u_p - u_d - float(dI_base)

    def step(self, setpoint, measurement):
        e = setpoint - measurement


        
        w_d = (-1 * measurement) if self.der_on_meas else (e)
        w_p = (-1 * measurement) if self.prop_on_meas else (e)

        u_p = self.kp * w_p
        
        u_d = self.k_u * self.d_prev + self.k_w * w_d - self.k_w * self.w_prev
        self.d_prev = u_d
        self.w_prev = w_d

        if self.integral_disc_method == "tustin":
            dI_base = 0.5 * self.ki * self.ts * (e + self.e_prev)
        elif self.integral_disc_method == "backeuler":
            dI_base = self.ki * self.ts * e
        else:
            dI_base = self.ki * self.ts * self.e_prev

        u_i = self.i_state + dI_base

        u_unsat = u_p + u_i + u_d
        u = max(self.min_output, min(self.max_output, u_unsat))

        if self.anti_windup == "backcalc":
            aw = self.kb_aw * (u - u_unsat)
            self.i_state = u_i + aw
        elif self.anti_windup == "clamping":
            upper = u_unsat >= self.max_output and e > 0.0
            lower = u_unsat <= self.min_output and e < 0.0
            if not (upper or lower):
                self.i_state = u_i
        else:
            self.i_state = u_i
            
        self.e_prev = e
        return u

    def reset(self):
        self.i_state = 0.0
        self.e_prev = 0.0
        self.d_prev = 0.0
        self.w_prev = 0.0
