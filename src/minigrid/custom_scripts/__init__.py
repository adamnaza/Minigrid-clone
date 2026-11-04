# Package-level registrations 
from .storage import get_storage_dir, PROJECT_ROOT

from gymnasium.envs.registration import register

TRAIN_ENV = "BabyAI-PickupNegation-v0"

register(id=TRAIN_ENV, entry_point="minigrid.custom_scripts.pickup_negation:PickupNegation")
