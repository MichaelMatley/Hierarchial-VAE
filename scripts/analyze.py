# Configuration for β-VAE sweep experiment

experiment:
  name: "beta_vae_sweep"
  seed: 42

# Base config (inherits from default)
base_config: "../default_config.yaml"

# Sweep over different beta values
sweep:
  parameter: "beta_schedule.max_beta"
  values: [0.1, 0.5, 1.0, 2.0, 5.0]

training:
  epochs: 50  # Shorter for sweep

beta_schedule:
  mode: "linear"
  warmup_epochs: 10
