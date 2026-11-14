import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from scipy.integrate import solve_ivp

# PARAMETERS AND DATA PREPARATION
alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 3.0

def f(t, X):
    x1, x2 = X

    dx1_dt = alpha * x1 - beta * x1 * x2
    dx2_dt = delta * x1 * x2 - gamma * x2
    return dx1_dt, dx2_dt

X0 = [1.0, 1.0]
t_span = [0.0, 5.0]
t_eval = np.linspace(0.0, 5.0, 50)
sol_real = solve_ivp(f, t_span, X0, t_eval=t_eval)

time = torch.tensor(t_eval, dtype=torch.float32)
X_real = torch.tensor(sol_real.y, dtype=torch.float32).T

meanX = X_real.mean(dim=0, keepdim=True)
stdX = X_real.std(dim=0, keepdim=True)
X_norm = (X_real - meanX) / stdX
X0_norm = X_norm[0]

# MODEL DEFINITION
class NeuralODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

    def forward(self, t, x):
        return self.net(x)

# LOAD PRETRAINED MODEL
model = NeuralODE()
weights = torch.load("lk_neuralODE.pt", weights_only=True)
model.load_state_dict(weights)

# VISUALIZATION
model.eval()
with torch.no_grad():
    X_pred = odeint(model, X0_norm, time).numpy()
    X_pred = meanX + stdX * X_pred

plt.figure(figsize=(10, 5))
plt.plot(t_eval, sol_real.y[0], 'k-', label='Presas (Real)', linewidth=2)
plt.plot(t_eval, X_pred[:, 0], 'r--', label='Presas (Neural ODE)', linewidth=2)

plt.plot(t_eval, sol_real.y[1], 'b-', label='Depredadores (Real)', linewidth=2)
plt.plot(t_eval, X_pred[:, 1], 'g--', label='Depredadores (Neural ODE)', linewidth=2)

plt.title("Neural ODE para Sistema No Lineal (Lotka-Volterra)")
plt.xlabel("Tiempo (t)")
plt.ylabel("Población")
plt.legend()
plt.grid(True)
plt.show()