"""
drone_crop_vision

A lightweight, edge-deployable plant disease classification package,
developed for Task 1 of the Advanced Robotics Automation and Computer
Vision module. Wraps a MobileNetV2 backbone fine-tuned on the PlantVillage
dataset, with augmentation intended to approximate drone-captured aerial
imaging conditions.

Modules are split by responsibility so that each stage (configuration,
data loading, model definition, training, evaluation, inference) can be
imported, tested and run independently, from either a plain Python
environment (VS Code, a terminal) or a notebook (Jupyter, Google Colab).
"""

__version__ = "1.0.0"
