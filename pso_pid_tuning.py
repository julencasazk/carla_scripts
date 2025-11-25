"""
This script is used to calculate Kp Ki and Kd parameters for a PID
controller, given a plant to control in closed-loop negative feedback
using Particle Swarm Optimization.

The cost function is calculated following the implmementation at:

        https://ieeexplore.ieee.org/document/8440927

Where:

    J1(X) = |SSE|   # Steady-State Error
    J2(X) = OS      # Overshoot
    J3(X) = Ts-Tr   # Settling time - Rise time ->  
    J4(X) = RVG     # Robustness for gain change
    J5(X) = ODJ     # Output disturbance rejection
    J6(X) = CEC     # Control Effort Limit


"""
import control as ctl
from PID import PID
import numpy as np

"""
Generic PSO algorithm implementation, to minimize function objective(x).

Params 
----------
objetive : function
    Function to minimize, should take a 1D numpy array as input.
bounds : list of tuples
    List of (min, max) tuples for each dimension.
num_particles : int
    Number of particles in the swarm.
max_it : int
    Maximum number of iterations.
w : float
    Inertia weight.
c1 : float
    Cognitive coefficient.
c2 : float
    Social coefficient.
seed : int or None
    Random seed for reproducibility.
verbose : bool
    If True, prints progress information.
    
Returns
-------
gbest_position : numpy array
    Best position found.
gbest_value : float
    Best objective value found.
history : list
    History of best objective values per iteration.

"""
def pso(objetive, bounds, num_particles=30, max_it=50, w=0.7, c1=1.5, c2=1.5, seed=None, verbose=True):
    
    
    if seed is None:
        np.random.seed(seed)
        
    dim = len(bounds)
    bounds_min = np.array([b[0] for b in bounds])
    bounds_max = np.array([b[1] for b in bounds])
    
    # Initialize particles and speeds
    particles = np.random.uniform(bounds_min, bounds_max, size=(num_particles, dim))
    velocities = np.random.uniform(-1, 1, size=(num_particles, dim))
    
    # Personal best
    pbest_positions = particles.copy()
    pbest_values = np.array([objetive(p) for p in particles])
    
    # Global best
    best_idx = np.argmin(pbest_values)
    gbest_position = pbest_positions[best_idx].copy()
    gbest_value = pbest_values[best_idx].copy()
    
    history = [gbest_value]
    
    if verbose:
        print(f"Iteration 0 - best cost: {gbest_value:.6f}")
        
        
    # Main loop
    for it in range(1, max_it + 1):
        for i in range(num_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            # Update speed
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - particles[i])
                + c2 * r2 * (gbest_position - particles[i])
            )
            # Update position
            particles[i] += velocities[i]
            
            # Apply boundries
            particles[i] = np.clip(particles[i], bounds_min, bounds_max)
            
            # Eval new cost
            J = objetive(particles[i])
            
            # Update personal and global bests
            if J < pbest_values[i]:
                pbest_values[i] = J
                pbest_positions[i] = particles[i].copy()
            
            if J < gbest_value:
                gbest_value = J
                gbest_position = particles[i].copy()
                
        history.append(gbest_value)
        if verbose:
            print(f"Iteration {it} - Best cost: {gbest_value:.6f}")
            
    return gbest_position, gbest_value, history

"""
Helper function to compute the robustness to variation in plant gain, or RVG

Scales the plant's transfer function with a certain gain
"""
def scale_plant(Gd, gain):
    num, den = ctl.tfdata(Gd)
    num = np.squeeze(num) * gain
    den = np.squeeze(den)
    return ctl.TransferFunction(num, den, Gd.dt)
            

"""
Closed loop simulation with discretized PID Class and a sample plant.
Plant must come in a discretized Gd, such as output from control.c2d()

Returns
-------
t: numpy array
    Time vector.
y: numpy array
    Output vector.
"""
def sim_closed_loop_pid(kp, ki, kd,
                        N, Ts,
                        Gd,
                        u_min, u_max, kb_aw,
                        t_final=5.0,
                        ref=1.0, disturb_time=20.0, disturb_value=5.0):
    
    
    pid = PID(kp, ki, kd, N, Ts, u_min, u_max, kb_aw, der_on_meas=False)
    
    # Sim time
    n_steps = int(t_final / Ts) + 1
    t = np.linspace(0.0, t_final, n_steps)
 
    u = np.zeros(n_steps)
    y = np.zeros(n_steps)


    
    # ========================
    # Change plant model BELOW
    # ========================
    num, den = ctl.tfdata(Gd)
    num = np.squeeze(num)
    den = np.squeeze(den)
    
    if den[0] != 1.0:
        num = num / den[0]
        den = den / den[0]
        
    # Order of denominator and numerator
    nb = len(num) - 1
    na = len(den) - 1
    
    for k in range(n_steps):
        # For calculating the output disturbance rejection or ODJ
        if disturb_time is not None and t[k] >= disturb_time:
            measured = y[k] + disturb_time
        else:
            measured = y[k]
            
        u[k] = pid.step(ref, 
                        measured if k < len(y) else y[-1]
                        )

        if k < n_steps - 1:
            
            y_next = 0.0
            
            for i in range(1, na + 1):
                idx_y = k + 1 - i
                if idx_y >= 0:
                    y_next -= den[i] * y[idx_y]
            
            for j in range(0, nb + 1):
                idx_u = k + 1 - j
                if idx_u >= 0:
                    y_next += num[j] * u[idx_u]
            
            y[k + 1] = y_next
            
    return t, y


"""
Computes metrics for cost function of PSO iterations
Parameters
----------
t : numpy array
    Time vector.
y : numpy array
    Output vector.
ref : float
    Reference value for step response.
tol : float
    Tolerance for settling time calculation.
    
Returns
-------
SSE : float
    Steady-State Error.
OS : float
    Overshoot percentage.
Tr : float
    Rise Time.
Ts : float
    Settling Time.

"""
def step_metrics(t, y, ref=1.0, tol=0.02):
    y = np.asarray(y)
    t = np.asarray(t)
    
    # Steady State Error
    SSE = abs(ref - y[-1])
    # Overshoot in %
    OS = max(0.0, (np.max(y) - ref) / ref * 100)
    
    # Rise Time
    y10 = 0.1 * ref
    y90 = 0.9 * ref
    try:
        idx_10 = np.where(y > y10)[0][0]
        idx_90 = np.where(y > y90)[0][0]
        Tr = t[idx_90] - t[idx_10]
    except IndexError:
        # Return extreme values so that the cost is big
        Tr = t[-1]        
        OS += 100.0
        
    lower = ref * (1 - tol)
    upper = ref * (1 + tol)
    Ts_ = t[-1]
    for i in range(len(t) - 1, -1, -1):
        if y[i] < lower or y[i] > upper:
            Ts_ = t[i + 1] if (i + 1) < len(t) else t[-1]
            break
        
    return SSE, OS, Tr, Ts_


"""
Computes the cost for a PSO iteration using
a PID controller (x) and a discrete plant (Gd)
"""
def pid_cost_custom_with_Gd(x,
                            Gd,
                            N, Ts,
                            u_min, u_max, kb_aw,
                            t_final=5.0,
                            ref=1.0,
                            w_sse=0.35,
                            w_os=0.5,
                            w_ts_tr =0.45):

    kp, ki, kd = x
    
    if kp < 0.0 or ki < 0.0 or kd < 0.0:
        return 1e9
   
   # Nominal value, without changing system gain 
    t_nom, y_nom = sim_closed_loop_pid(kp, ki, kd,
                               N, Ts,
                               Gd,
                               u_min, u_max, kb_aw,
                               t_final, ref,
                               disturb_time=None,
                               disturb_value=None
    )
    
    SSE_nom, OS_nom, Tr_nom, Ts__nom = step_metrics(t_nom, y_nom, ref)
   
    # The same, but changing system gain lower 
    Gd_low = scale_plant(Gd, 0.7)
    t_low, y_low = sim_closed_loop_pid(kp, ki, kd,
                                N, Ts,
                                Gd_low,
                                u_min, u_max, kb_aw,
                                t_final, ref,
                                disturb_time=None,
                                disturb_value=None)
    
    SSE_low, OS_low, Tr_low, Ts__low = step_metrics(t_low, y_low, ref)
    
    # The same, but increasing the system gain
    Gd_high = scale_plant(Gd, 1.3) 
    t_high, y_high = sim_closed_loop_pid(kp, ki, kd,
                                         N, Ts,
                                         Gd_high,
                                         u_min, u_max, kb_aw,
                                         t_final, ref, 
                                         disturb_time=None, 
                                         disturb_value=None)
    SSE_high, OS_high, Tr_high, Ts__high = step_metrics(t_high, y_high, ref)
    
    # Reference values for each magnitude
    # Change for acceptable values
    # ===================================
    SSE_ref = 1.0 # 1m/s deviation
    OS_ref = 10.0 # 10% overshoot deviation
    Ts__ref = 5.0 # 5s settle time deviation
    # Weights for cost function 
    w_d_sse = 0.5
    w_d_os = 1.5
    w_d_ts = 1.0
    
    # Compute the other cost deltas
    dSSE = max(SSE_low - SSE_nom, SSE_high - SSE_nom, 0.0) / SSE_ref
    dOS = max(OS_low - OS_nom, OS_high - OS_nom, 0.0) / OS_ref
    dTs_ = max(Ts__low - Ts__nom, Ts__high - Ts__nom, 0.0) / Ts__ref
    
    RVG = w_d_sse*dSSE + w_d_os*dOS + w_d_ts*dTs_ 
    
    
    t_dist, y_dist = sim_closed_loop_pid(kp, ki, kd,
                               N, Ts,
                               Gd,
                               u_min, u_max, kb_aw,
                               t_final, ref,
                               disturb_time=t_final*0.6, # After the middle of the test
                               disturb_value=ref*0.1     # 10% of setpoint, idk, winging it
    )
    
    # Error after system output disturbance
    mask = t_dist>= t_final*0.6
    e_dist = ref - y_dist[mask]
    
    # Calculate integral of quad error (ISE)
    J5_ODJ = np.trapezoid(e_dist**2, t_dist[mask])


    
    # TODO Implement the rest of the cost
    J1 = SSE_nom
    J2 = OS_nom
    J3 = max(0.0, Ts__nom - Tr_nom) # Seems like a heuristic cost function, don't know if it means anything physical
    J4 = RVG
    
    
    
    
    
    J = w_sse*J1 + w_os*J2 + w_ts_tr*J3
    return J

"""
Computes cost related to frequency domain for a PSO iteration.
It needs the current iterations's gains to reconstruct a PID.
"""
def freq_metrics(kp, ki, kd, Gc,
                 w_min=1e-3, w_max=10.0, n_points=500,
                 B_dB=-20.0,
                 R_dB=10.0):
    
    
def main():
    
    # Plant params
    k = 0.156
    wn = 0.396
    zeta = 0.661
    Td = 0.146

    # Simulation / sampling
    N_filt = 20.0
    Ts = 0.01
    u_min = 0.0
    u_max = 100.0
    kb_aw = 1.0
    T_sim = 80.0
    N_steps = int(T_sim / Ts)

    # Continuous-time plant without delay
    s = ctl.TransferFunction.s
    G0 = k / (s**2 + 2*zeta*wn*s + wn**2)

    # Delay as Pade(1)
    num_delay, den_delay = ctl.pade(Td, 1)
    H_delay = ctl.tf(num_delay, den_delay)

    # Full continuous plant (with Pade delay), then discretize
    Gc = G0 * H_delay
    Gd = ctl.c2d(Gc, Ts, method='tustin')
    
    def objective_pid(x):
        return pid_cost_custom_with_Gd(
            x, Gd=Gd,
            N=N_filt, Ts=Ts,
            u_min=u_min, u_max=u_max, kb_aw=kb_aw,
            t_final=T_sim, ref=50.0
        )
        
    # PSO bounds for Kp, Ki, Kd
    bounds = [(0.0, 100.0),   # Kp
              (0.0, 100.0),    # Ki
              (0.0, 100.0)]   # Kd
    
    best_x, best_J, history = pso(objective_pid,
                                  bounds,
                                  num_particles=50,
                                  max_it=100,
                                  w=0.7,
                                  c1=1.5,
                                  c2=1.5,
                                  seed=0,
                                  verbose=True)
    
    print(f"Best X: {best_x}\nBest J: {best_J}")
    

    
    

if __name__ == "__main__":
    main()
