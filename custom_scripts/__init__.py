"""Project environments. Import this package to register them with Gymnasium."""

from gymnasium.envs.registration import register

TRAIN_ENV = "BabyAI-PickupNegation-v0"
TEST_ENV = "BabyAI-PickupNegation-Test-v0"

register(id=TRAIN_ENV, entry_point="custom_scripts.pickup_negation:PickupNegation")
register(
    id=TEST_ENV,
    entry_point="custom_scripts.pickup_negation:PickupNegation",
    kwargs={"split": "test"},
)
