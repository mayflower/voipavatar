"""
Persona configuration loader.

This module provides utilities for loading and managing persona configurations
from the personas.yaml file.

Classes:
    PersonaConfig: Dataclass holding all configuration for a persona

Functions:
    load_persona: Load a specific persona by name from YAML config
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


class PersonaNotFoundError(Exception):
    """Raised when a requested persona is not found in the configuration."""


@dataclass
class PersonaConfig:
    """
    Configuration for a single persona/avatar.

    Attributes:
        name: Internal key name for the persona
        display_name: Human-readable name for display
        video_path: Path to the avatar reference video file
        inference_config: Path to MuseTalk inference config YAML
        bbox_shift: Vertical shift for face bounding box (pixels)
    """

    name: str
    display_name: str
    video_path: str
    inference_config: str
    bbox_shift: int = 0


def _get_personas_yaml_path() -> Path:
    """Get the path to the personas.yaml file."""
    # Look for personas.yaml in the same directory as this module
    module_dir = Path(__file__).parent
    yaml_path = module_dir / "personas.yaml"

    if yaml_path.exists():
        return yaml_path

    # Fall back to environment variable if set
    env_path = os.environ.get("PERSONAS_YAML_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Check in /app/service (container location)
    container_path = Path("/app/service/personas.yaml")
    if container_path.exists():
        return container_path

    raise FileNotFoundError(
        f"personas.yaml not found. Checked: {yaml_path}, {container_path}"
    )


def load_all_personas() -> dict[str, PersonaConfig]:
    """
    Load all personas from the configuration file.

    Returns:
        Dictionary mapping persona names to PersonaConfig objects

    Raises:
        FileNotFoundError: If personas.yaml is not found
    """
    yaml_path = _get_personas_yaml_path()

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    personas = {}
    for name, data in config.items():
        personas[name] = PersonaConfig(
            name=name,
            display_name=data.get("display_name", name),
            video_path=data["video_path"],
            inference_config=data["inference_config"],
            bbox_shift=data.get("bbox_shift", 0),
        )

    return personas


def load_persona(persona_name: str) -> PersonaConfig:
    """
    Load a specific persona by name from the YAML config.

    Args:
        persona_name: The key name of the persona to load (e.g., "default", "news_anchor")

    Returns:
        PersonaConfig object with the persona's settings

    Raises:
        PersonaNotFoundError: If the persona name is not found
        FileNotFoundError: If personas.yaml is not found
    """
    personas = load_all_personas()

    if persona_name not in personas:
        available = ", ".join(personas.keys())
        raise PersonaNotFoundError(
            f"Persona '{persona_name}' not found. Available personas: {available}"
        )

    return personas[persona_name]


def list_persona_names() -> list[str]:
    """
    Get a list of all available persona names.

    Returns:
        List of persona key names
    """
    return list(load_all_personas().keys())
