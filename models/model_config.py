# Configuration globale pour les dimensions des modèles
DEFAULT_NUM_VERTICES = 1000
MIN_VERTICES = 100
MAX_VERTICES = 5000

# Configuration pour l'entraînement
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Configuration pour la génération
MAX_GENERATION_ATTEMPTS = 10
NOISE_DIMENSION = 100

# Configuration pour l'évaluation
METRICS_OUTPUT_FILE = "./data/metrics.json"