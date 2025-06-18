import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Red simple totalmente conectada
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
        return self.net(t) * self.scale 

# Derivada usando autograd
def derivada_N(modelo, t):
    t.requires_grad = True
    N = modelo(t)
    dN_dt = torch.autograd.grad(N, t, torch.ones_like(N), create_graph=True)[0]
    return dN_dt

# Función de pérdida basada en la ecuación diferencial
def loss_func(modelo, t):
    N = modelo(t)
    dN_dt = derivada_N(modelo, t)
    edo_residual = dN_dt - (30 - 0.1 * N)
    loss_edo = torch.mean(edo_residual**2)

    loss_ini = (modelo(torch.tensor([[0.0]])) - 100.0)**2  # N(0) = 100
    return loss_edo + loss_ini

# Entrenamiento
modelo = PINN()
opt = torch.optim.Adam(modelo.parameters(), lr=1e-3) # saltos mas pequeños, mas lento, pero preciso
t_train = torch.linspace(0, 60, 100).reshape(-1, 1)

for epoch in range(5000):
    loss = loss_func(modelo, t_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")

# Graficar resultados
t_test = torch.linspace(0, 60, 200).reshape(-1, 1)
N_pred = modelo(t_test).detach().numpy()

N_analitico = 300 - 200 * np.exp(-0.1 * t_test.numpy())

plt.plot(t_test, N_pred, label='PINN')
plt.plot(t_test, N_analitico, '--', label='Solución Analítica')
plt.title("Solución mejorada")
plt.xlabel("Tiempo (min)")
plt.ylabel("Número de autos")
plt.grid()
plt.legend()
plt.show()