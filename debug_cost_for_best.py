import numpy as np
import control as ctl
from pso_pid_tuning import pid_cost_custom_with_Gd, sim_closed_loop_pid, step_metrics

def main():
    Ts = 0.01
    Td = 0.15
    s = ctl.TransferFunction.s
    G0 = (12.68) / (s**2 + 1.076*s + 0.2744)
    num_delay, den_delay = ctl.pade(Td, 1)
    H_delay = ctl.tf(num_delay, den_delay)
    Gc = G0 * H_delay
    Gd = ctl.c2d(Gc, Ts, method='tustin')

    N_filt = 20.0
    u_min, u_max = 0.0, 1.0
    kb_aw = 1.0
    T_sim = 80.0
    ref = 33.33

    x_bad = np.array([1.05730764, 1.3201751, 5.0])

    # First: raw metrics with same sim/metrics used in cost
    t_nom, y_nom, u_nom = sim_closed_loop_pid(
        x_bad[0], x_bad[1], x_bad[2],
        N_filt, Ts,
        Gd,
        u_min, u_max, kb_aw,
        t_final=T_sim, ref=ref,
        disturb_time=None,
        disturb_value=None
    )
    SSE_nom, OS_nom, Tr_nom, Ts__nom = step_metrics(t_nom, y_nom, ref)
    print("Raw metrics for bad PID:")
    print(f"  SSE_nom = {SSE_nom:.3f}")
    print(f"  OS_nom  = {OS_nom:.3f} %")
    print(f"  Tr_nom  = {Tr_nom:.3f} s")
    print(f"  Ts_nom  = {Ts__nom:.3f} s")

    # Second: full cost decomposition
    J, info = pid_cost_custom_with_Gd(
        x_bad,
        Gd=Gd,
        N=N_filt, Ts=Ts,
        u_min=u_min, u_max=u_max, kb_aw=kb_aw,
        t_final=T_sim, ref=ref,
        bounded=True
    )

    print("\nFrom cost function:")
    print(f"  J1_SSE_norm = {info['J1_SSE_norm']:.6f}")
    print(f"  J2_OS_norm  = {info['J2_OS_norm']:.6f}")
    print(f"  J3_Ts_Tr    = {info['J3_Ts_Tr_norm']:.6f}")
    print(f"  Total_J     = {info['Total_J']:.6f}")

if __name__ == '__main__':
    main()
