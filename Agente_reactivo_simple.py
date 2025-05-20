class SimpleReactiveAgent:
    def __init__(self, name):
        self.name = name

    def interpret_input(self, world_description):
        print("\nInterpretacion de la entrada:")
        if world_description["humedad"] < 30: 
            print("- Humedad baja")
        else:
            print("- Humedad suficiente")
        
        if world_description["lluvia"]:
            print("- Lluvia pronosticada")
        else:
            print("- Sin lluvia pronosticada")

        if world_description["temperatura"] < 15:
            print("- Temperatura baja")
        elif world_description["temperatura"] > 30:
            print("- Temperatura alta")
        else:
            print("- Temperatura moderada")

        return world_description

    def rule_coincidence(self, rules, state):
        for rule in rules:
            if rule["condicion"](state):
                return rule
        return None

    def rule_action(self, rule):
        return rule["accion"]
    
    def actuator(self, action):
        print("\nActuador:")
        if action == "Activar riego":
            print("- Válvula abierta \n- Aspersores activados")
        elif action == "Desactivar riego":
            print("- Válvula cerrada \n- Aspersores desactivados")
        else:
            print("\nActuador: Acción desconocida o no necesaria.")

    def act(self, rules, world_description):
        state = self.interpret_input(world_description)
        rule = self.rule_coincidence(rules, state)

        if rule:
            action = self.rule_action(rule)
        else:
            action = "Sin accion"

        self.actuator(action)

        return action

if __name__ == "__main__":
    # Definicion del agente
    irrigation_agent_rules = [
        {
            "condicion": lambda estado: estado["humedad"] < 30 and not estado["lluvia"],
            "accion": "Activar riego"
        },
        {
            "condicion": lambda estado: estado["humedad"] >= 30 or estado["lluvia"],
            "accion": "Desactivar riego"
        }
    ]
    
    # Creacion del agente
    agent_name = "Agente Reactivo Simple de Riego Automático"
    irrigation_agent = SimpleReactiveAgent(agent_name)

    world_description = {
        "humedad": 25,     # %
        "lluvia": False,   # Pronóstico
        "temperatura": 18  # °C
    }

    # Ejecucion del agente
    print("-------------------------")
    print(f"Ejecutando el {agent_name}...")
    action = irrigation_agent.act(irrigation_agent_rules, world_description)
    print(f"\nEl agente decidió: \n- {action}")
    print("-------------------------")
