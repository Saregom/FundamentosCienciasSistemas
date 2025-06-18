import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Red neuronal con normalización en la salida
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.scale = 300.0  # Factor de normalización

    def forward(self, t):
        return self.net(t) * self.scale  # Desnormaliza la salida

# Derivada
def derivada_N(modelo, t):
    t.requires_grad = True
    N = modelo(t)
    dN_dt = torch.autograd.grad(N, t, torch.ones_like(N), create_graph=True)[0]
    return dN_dt

# Función de pérdida con ponderación mejorada
def loss_func(modelo, t):
    N = modelo(t)
    dN_dt = derivada_N(modelo, t)
    edo_residual = dN_dt - (30 - 0.1 * N)
    loss_edo = torch.mean(edo_residual**2)
    
    # Condición inicial reforzada (varios puntos cerca de t=0)
    t_ini = torch.linspace(0, 0.5, 10).reshape(-1, 1)
    N_ini = modelo(t_ini)
    loss_ini = torch.mean((N_ini - 100.0)**2)
    
    return loss_edo + 100 * loss_ini  # Peso mayor a la condición inicial

# Entrenamiento con LR más alto
modelo = PINN()
opt = torch.optim.Adam(modelo.parameters(), lr=1e-2)  
t_train = torch.linspace(0, 60, 200).reshape(-1, 1)

for epoch in range(10000):
    loss = loss_func(modelo, t_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# Solución y analítica para comparación
t_test = torch.linspace(0, 60, 200).reshape(-1, 1)
N_pred = modelo(t_test).detach().numpy()

N_analitico = 300 - 200 * np.exp(-0.1 * t_test.numpy())

# Gráfica
plt.plot(t_test, N_pred, label='PINN')
plt.plot(t_test, N_analitico, '--', label='Solución Analítica')
plt.title("Comparación PINN vs Solución Exacta")
plt.xlabel("Tiempo (min)")
plt.ylabel("Número de autos (N)")
plt.grid()
plt.legend()
plt.show()

