import os
import hydragnn

filepath = os.path.join(os.path.dirname(__file__), "ising_model.json")
hydragnn.run_training(filepath)
