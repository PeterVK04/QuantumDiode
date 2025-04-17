import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import time

# Parameters
M = 11; J = 1.0
lams = np.linspace(0, 1, 21)
taus = np.logspace(1, 3, 20)
time_steps = 200  # increase for accuracy

# The new propagator‐based fidelity function
def compute_fidelities(M, J, lam, total_tau, time_steps):
    N = 2*M - 1
    # site map
    st = {}
    idx = 0
    for m in range(1, M+1):
        st[('A',m)] = idx; idx+=1
        if m<M:
            st[('B',m)] = idx; idx+=1

    def phi(t): return (t/total_tau)*np.pi
    def J1(t):  return J*(1 - lam*np.cos(phi(t)))
    def J2(t):  return J*(1 + lam*np.cos(phi(t)))

    # Precompute unitaries
    ts = np.linspace(0, total_tau, time_steps+1)
    dt = ts[1]-ts[0]
    Us = []
    for k in range(time_steps):
        tmid = 0.5*(ts[k]+ts[k+1])
        j1, j2 = J1(tmid), J2(tmid)
        H = np.zeros((N,N), complex)
        for m in range(1,M):
            a=st[('A',m)]; b=st[('B',m)]
            H[a,b] = H[b,a] = -j1
            a2=st[('A',m+1)]
            H[a2,b]=H[b,a2] = -j2
        Us.append(( -1j * qt.Qobj(H) * dt ).expm())

    # Propagate left→right
    psi = qt.basis(N, st[('A',1)])
    for U in Us: psi = U*psi
    F_l = abs( psi[ st[('A',M)] ] )**2

    # Propagate right→left
    psi = qt.basis(N, st[('A',M)])
    for U in Us: psi = U*psi
    F_r = abs( psi[ st[('A',1)] ] )**2

    return F_l, F_r

# Sweep
F_l = np.zeros((len(lams), len(taus)))
F_r = np.zeros_like(F_l)
t0 = time.time()
for i, lam in enumerate(lams):
    for j, Tau in enumerate(taus):
        F_l[i,j], F_r[i,j] = compute_fidelities(M, J, lam, Tau, time_steps)
    print(f"λ={lam:.2f} done")
print("Sweep time:", time.time()-t0)


# ───── Plot with pcolormesh + log‐scale y ─────
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

# create mesh for pcolormesh
X, Y = np.meshgrid(lams, taus)

# note: F_l.T is shape (len(taus), len(lams))
pcm1 = ax1.pcolormesh(X, Y, F_l.T, cmap='viridis', vmin=0, vmax=1, shading='auto')
ax1.set_xscale('linear')
ax1.set_yscale('log')
ax1.set_yticks([10, 100, 1000])
ax1.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel('$\\lambda$')
ax1.set_ylabel('$\\tau$')
ax1.set_title('Fidelity $F_l$ (A₁→A_M)')
fig.colorbar(pcm1, ax=ax1, label='$F_l$')

pcm2 = ax2.pcolormesh(X, Y, F_r.T, cmap='viridis', vmin=0, vmax=1, shading='auto')
ax2.set_xscale('linear')
ax2.set_yscale('log')
ax2.set_yticks([10, 100, 1000])
ax2.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax2.set_xlabel('$\\lambda$')
ax2.set_ylabel('$\\tau$')
ax2.set_title('Fidelity $F_r$ (A_M→A₁)')
fig.colorbar(pcm2, ax=ax2, label='$F_r$')

plt.tight_layout()
plt.show()
