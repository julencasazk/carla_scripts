"""
Plots step responses for selected PID gains and prints speed metrics.

Outputs
  - Matplotlib plots and console metrics.

Run (example)
  python3 tuning/pso/debug_speed_metrics.py

# filepath: /home/jcazk/carla_scripts/debug_speed_metrics_plot.py
"""

import numpy as np
import control as ctl
import matplotlib.pyplot as plt
from lib.pso_pid_tuning import sim_closed_loop_pid, step_metrics

def run_case(kp, ki, kd, label, ax_y, ax_u, color, N_filt=20.0, kb_aw=1.0):
    Ts = 0.01
    Td = 0.15
    s = ctl.TransferFunction.s
    '''
    G0 = (12.68) / (s**2 + 1.076*s + 0.2744)
    num_delay, den_delay = ctl.pade(Td, 1)
    H_delay = ctl.tf(num_delay, den_delay)
    Gc = G0 * H_delay
    Gd = ctl.c2d(Gc, Ts, method='tustin')
    '''

    G0 = (8.7129*s + 0.05263) / (s**2 + 0.1953*s + 0.0001874) 
    Gd = ctl.c2d(G0, Ts, method='tustin')
    
    print(Gd)

    u_min, u_max = 0.0, 1.0          # throttle in [0,1]
    T_sim = 80.0
    ref = 33.33

    t, y, u = sim_closed_loop_pid(
        kp, ki, kd,
        N_filt, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final=T_sim, ref=ref,
        disturb_time=None,
        disturb_value=None
    )

    SSE, OS, Tr, Ts_ = step_metrics(t, y, ref)
    TsTr = max(0.0, Ts_ - Tr)

    print(f"{label}:")
    print(f"  Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
    print(f"  Tr   = {Tr:.2f} s")
    print(f"  Ts   = {Ts_:.2f} s")
    print(f"  Ts-Tr= {TsTr:.2f} s")
    print(f"  OS   = {OS:.1f} %")
    print(f"  y_end= {y[-1]:.2f}")
    print("")

    # Plot y
    ax_y.plot(t, y, label=f"{label}", color=color)
    ax_y.axhline(ref, ls='--', color='k', lw=0.8)

    # Plot u
    ax_u.plot(t, u, label=f"{label}", color=color)

def main():
    fig, (ax_y, ax_u) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    #kp = 0.0694358
    #ki = 0.00758438
    #kd = 0.86238791
    
    #run_case(kp, ki, kd, "Dampened", ax_y, ax_u, "C1")
    
    #kp = 0.63585695
    #ki = 0.85758202
    #kd = 3.63651481

    #run_case(kp, ki, kd, "Aggresive", ax_y, ax_u, "C3")
    
    '''
    # New test with the simulation method fixed,
    # should be truer to the real plant
    kp = 0.21857731
    ki = 0.05122361
    kd = 5.0
    run_case(kp, ki, kd, "Fixed method", ax_y, ax_u, "C4")
    
    # Ranked cost scaling, seems catastrophic
    kp = 1.74258599
    ki = 0.10109199
    kd = 4.16309923
    #run_case(kp, ki, kd, "Ranked cost scaling", ax_y, ax_u, "C5")
    '''
    
    '''
    # Normalized cost scaling, scaling with max and min each iteration
    kp = 0.78735842
    ki = 0.48564073
    kp = 6.38541476
    #run_case(kp, ki, kd, "Normalized scaling 1", ax_y, ax_u, "C6")
    
    
    # Normalized cost scaling, scaling with max and min each iteration ANOTHER ONE
    kp = 1.55804038
    ki = 0.49750068
    kp = 5.31105199
    #run_case(kp, ki, kd, "Normalized scaling 2", ax_y, ax_u, "C7")
    '''
    '''
    # Normal PID tuning, but with no target theshold, minimum 0.0
    kp = 0.60132401
    ki = 0.25121954
    kd = 7.73220397
    run_case(kp, ki, kd, "Normal PID without _target", ax_y, ax_u, "C8")
    '''
    
    # Normalized with offline metrics (median InterQuartile Range), new method in pid_tuning_offline_normalize.py
    #kp = 0.60868111
    #ki = 0.19973763
    #kd = 10.0
    #run_case(kp, ki, kd, "Offline normalization (median IQR)", ax_y, ax_u, "C9")

    ''' 
    # Normalized with Offline metrics, du_rms fixation, multiprocessing
    kp = 0.47446807
    ki = 0.15472707
    kd = 7.83353195
    run_case(kp, ki, kd, "Normal (median, IQR) du fixation", ax_y, ax_u, "C11")
    '''
    
    # Multiprocessing normal Kp, Ki, Kd tuning, with _target = 0.
    #kp = 0.60988305
    #ki = 0.19988017
    #kd = 9.99136697
    # seed 7077
    #run_case(kp, ki, kd, "Fixation on CEC", ax_y, ax_u, "C12")


    # Multiprocessing with Kp Ki N and k anti windup 
    '''
    kp = 3.70359920e-01
    ki = 1.09943679e-02
    kd = 5.40267246e+00
    N = 2.98547541e+01
    kaw = 1.79594124e-01
    run_case(kp, ki, kd, "NKaw", ax_y, ax_u, "C13", N, kaw)
    '''

    # Another try, but this time, with more "normal" weights
    # This is the MOST AGGRESIVE but without Overshoot for NOW
    #kp = 0.77658553
    #ki = 0.84037437
    #kd = 10.0
    #N = 39.34907758
    #kaw = 1.0
    #run_case(kp, ki, kd, "Including N on PSO", ax_y, ax_u, "C14", N, kaw)
    
    # ODJ changed from integral of the error^2 to the mean error^2. Seems to have yielded more reasonable response???
    #kp = 0.21008327
    #ki = 0.06662786
    #kd = 6.61821463
    #N  = 37.98325395
    #kaw = 1.0
    #run_case(kp, ki, kd, "Mean ODJ instead of integral", ax_y, ax_u, "C15", N, kaw)
    
    #kp =  0.55856216
    #ki = 0.1824219
    #kd = 6.71382146
    #N = 14.87180254
    #kaw = 1.0
    #run_case(kp, ki, kd, "Anothaone", ax_y, ax_u, "C16", N, kaw)
    
    #kp =  0.73728434
    #ki =  0.7365131
    #kd =  6.94588313
    #N = 29.16005392 
    #run_case(kp, ki, kd, "Aggresive2", ax_y, ax_u, "C17", N, kaw)

    #kp = 2.59397071e-01
    #ki = 1.27381733e-01
    #kd = 6.21160744e-03
    #N = 15.0
    #run_case(kp, ki, kd, "New model, same weights", ax_y, ax_u, "C18", N, kaw)

    kp = 8.85
    ki = 0.005
    kd = 0.00125
    N = 15.0
    kaw = 1.0
    run_case(kp, ki, kd, "Ziegler-Nichols", ax_y, ax_u, "C18", N, kaw)

    ax_y.set_ylabel("y [m/s]")
    ax_y.set_title("Closed-loop step responses (same sim as cost)")
    ax_y.grid(True)
    ax_y.legend()

    ax_u.set_ylabel("u (throttle)")
    ax_u.set_xlabel("t [s]")
    ax_u.grid(True)
    ax_u.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
