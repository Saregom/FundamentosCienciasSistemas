class ModelBasedReactiveAgent:
    def __init__(self, name):
        self.name = name
        self.state = {"posicion": (0, 0), "mapa": {}}

    def update_state(self, perception):
        x, y = perception["posicion"]
        suciedad = perception["suciedad"]
        obstaculo = perception["obstaculo"]

        print("\nEstado actual (Mapa):")
        if not self.state["mapa"]:
            print("- Mapa vacío")
        else:
            for pos, info in self.state["mapa"].items():
                print(f"-Posición {pos}: \n\t-Suciedad: {info['suciedad']} \n\t-Obsataculo: {info['obstaculo']}")

        print(f"\nPercepcion recibida:")
        print(f"- Posición actual: {x}, {y}")
        print(f"- Suciedad: {'Si' if suciedad else 'No'}")
        print(f"- Obsatuculos: {'Si' if obstaculo else 'No'}")

        pos = perception["posicion"]
        self.state["posicion"] = pos
        self.state["mapa"][pos] = {
            "suciedad": perception["suciedad"],
            "obstaculo": perception["obstaculo"]
        }
        
        return self.state

    def rule_coincidence(self, rules, state):
        for rule in rules:
            if rule["condicion"](state):
                return rule
        return None

    def rule_action(self, rule):
        return rule["accion"]

    def actuator(self, action):
        print("\nActuador:")
        if action == "Limpiar":
            print("- Activando aspiradora")
        elif action == "Mover":
            print("- Activando motores de desplazamiento")
        else:
            print("- Acción desconocida o innecesaria")

    def act(self, rules, perception):
        state = self.update_state(perception)
        rule = self.rule_coincidence(rules, state)

        if rule:
            action = self.rule_action(rule)
        else:
            action = "Sin acción"

        self.actuator(action)
        return action

if __name__ == "__main__":
    # Definicion del agente
    cleaning_rules = [
        {
            "condicion": lambda estado: 
                            estado["mapa"].get(estado["posicion"], {}).get("suciedad", False)
                            and not estado["mapa"].get(estado["posicion"], {}).get("obstaculo", False),
            "accion": "Limpiar"
        },
        {
            "condicion": lambda estado: 
                            not estado["mapa"].get(estado["posicion"], {}).get("suciedad", False) 
                            or estado["mapa"].get(estado["posicion"], {}).get("obstaculo", False),
            "accion": "Mover"
        }
    ]

    # Creacion del agente
    agent_name = "Agente Reactivo Basado en Modelos: Robot de Limpieza"
    cleaning_agent = ModelBasedReactiveAgent(agent_name)

    # Simulación de percepciones
    percepciones = [
        {"posicion": (0, 0), "suciedad": True, "obstaculo": False},
        {"posicion": (0, 1), "suciedad": False, "obstaculo": False},
        {"posicion": (1, 0), "suciedad": True, "obstaculo": False},
        {"posicion": (1, 1), "suciedad": False, "obstaculo": True}
    ]

    # Ejecucion del agente
    print("-------------------------")
    print(f"Ejecutando el {agent_name}...")
    for i, percepcion in enumerate(percepciones):
        print("\n-------------------------")
        print(f"Ciclo {i+1}:")
        accion = cleaning_agent.act(cleaning_rules, percepcion)
        print(f"\nEl agente decidió: \n- {accion}")
