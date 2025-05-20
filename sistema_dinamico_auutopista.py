import numpy as np

class TrafficControlSystem:
    def __init__(self, name, N0=100, t=1):
        self.name = name
        self.state = {
            "autos": N0,
            "historial": [N0]
        }
        self.r_in = 30 # tasa de entrada
        self.k = 0.1 # coeficiente de salida proporcional
        self.t = t # paso de tiempo (minutos)

    def calcular_autos(self):
        # Ecuacion del sistema: 300-200e^(-0.1t)
        return 300 - 200 * np.exp(-0.1*self.t)

    def update_state(self):
        N_nuevo = self.calcular_autos()
        self.state["autos"] = N_nuevo
        self.state["historial"].append(N_nuevo)
        print(f"\n[Modelo dinámico]\n- N({self.t}): {int(N_nuevo)}")
        self.t += 1

    def rule_coincidence(self, rules, state):
        for rule in rules:
            if rule["condicion"](state):
                return rule
        return None

    def rule_action(self, rule):
        return rule["accion"]

    def actuator(self, action):
        print("\n[Actuador activado]")
        if action == "Entrada libre":
            print("- Entrada libre, sin restricciones.")
        elif action == "Reducir entrada":
            print("- Ubicando personal para reducir el flujo.")
        elif action == "Cerrar acceso":
            print("- Cerrando entradas temporalmente.")
        else:
            print("- No se requiere accion especifica.")

    def act(self, rules):
        self.update_state()
        rule = self.rule_coincidence(rules, self.state)

        if rule:
            action = self.rule_action(rule)
        else:
            action = "Sin acción"

        self.actuator(action)
        return action
    
if __name__ == "__main__":
    # Definicion de reglas
    traffic_rules = [
        {
            "condicion": lambda estado: estado["autos"] < 200,
            "accion": "Entrada libre"
        },
        {
            "condicion": lambda estado: 200 <= estado["autos"] < 280,
            "accion": "Reducir entrada"
        },
        {
            "condicion": lambda estado: estado["autos"] >= 280,
            "accion": "Cerrar acceso"
        }
    ]

    # Creacion del sistema
    system_name = "Sistema de Control de Tráfico"
    traffic_system = TrafficControlSystem(system_name, 100, 1)

    print("-------------------------------")
    print(f"Ejecutando el {system_name}...\n")

    ciclos = 30  # cantidad de ciclos de simulación (media hora)

    # Ejecucion del sistema
    for i in range(ciclos):
        print(f"\n--- Ciclo {i+1} ---")
        action = traffic_system.act(traffic_rules)
        print(f"\n[Acción tomada]\n- {action}")
