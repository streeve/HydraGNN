import os
import hydragnn

filepath = os.path.join(os.path.dirname(__file__), "fept_single.json")
hydragnn.load_data_and_run_training(filepath)
