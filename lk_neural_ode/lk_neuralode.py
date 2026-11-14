import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from scipy.integrate import solve_ivp
import tqdm

# PARAMETERS AND DATA PREPARATION
alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 3.0

def f(t, X):
    x1, x2 = X

    dx1_dt = alpha * x1 - beta * x1 * x2
    dx2_dt = delta * x1 * x2 - gamma * x2
    return dx1_dt, dx2_dt

X0 = [1.0, 1.0]
t_span = [0.0, 10.0]
t_eval = np.linspace(0.0, 10.0, 200)
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

# TRAINING
model = NeuralODE()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

epochs = 2000
model.train()
progress_bar = tqdm.tqdm(range(epochs), desc=f"Starting training...")
min_loss = 1e6
for epoch in progress_bar:
    X_pred = odeint(model, X0_norm, time)

    optimizer.zero_grad()
    loss = torch.mean((X_pred - X_norm)**2)
    loss.backward()
    optimizer.step()

    current_loss = loss.item()

    if current_loss < min_loss:
        torch.save(model.state_dict(), "lk_neuralODE.pt")
        min_loss = current_loss

    if epoch % 50 == 0:
        progress_bar.set_postfix(loss=f"{current_loss:.6f}", min_loss=f"{min_loss:.6f}")

print("TRAINING SUCCESS!")