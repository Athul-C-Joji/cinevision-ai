"""
src/script_planner/prompt_templates.py

Prompt text for the Script Planner (Module 3). This module is LLM
orchestration only -- nothing here is trained.

The allowed labels come straight from the Module 1 vocabularies
(CLASS_VOCAB and MOVEMENT_CLASS_VOCAB), so the LLM's suggestions use the
same words the Video Analyzer produces.
"""

from src.data.label_encoding import CLASS_VOCAB
from src.data.movement_label_encoding import MOVEMENT_CLASS_VOCAB

MOVEMENT_KEY = "Movement"


def vocab_as_dict():
    """{dimension name: [allowed labels]} for the 7 static dimensions + Movement."""
    vocab = {dim: list(classes) for dim, classes in CLASS_VOCAB.items()}
    vocab[MOVEMENT_KEY] = list(MOVEMENT_CLASS_VOCAB)
    return vocab


def build_vocab_text():
    lines = []
    for dim, classes in vocab_as_dict().items():
        lines.append(f"- {dim}: " + " | ".join(classes))
    return "\n".join(lines)


def build_json_example():
    """Shows the exact reply shape, with every dimension key present."""
    dims = list(vocab_as_dict().keys())
    labels = ", ".join(f'"{d}": ["..."]' for d in dims)
    return (
        '{"shots": [{"shot_number": 1, '
        '"description": "one sentence: what the audience sees", '
        '"purpose": "one sentence: why this shot serves the scene", '
        f'"labels": {{{labels}}}}}]}}'
    )


SYSTEM_TEMPLATE = """You are a cinematography assistant. You help plan shots for scenes that have not been filmed yet.

You will be given ONE scene from a script. Suggest a shot list for it.

Rules:
1. For every shot, choose labels ONLY from the allowed vocabulary below, spelled exactly as written. Use one label per dimension normally. Use two only when both genuinely apply to the same shot.
2. Do not invent story events, characters or dialogue that are not in the scene text.
3. "description" is one sentence about what the audience sees. "purpose" is one sentence about why the shot serves the scene.
4. These are creative suggestions, not the only correct choices.
5. Reply with a single JSON object and nothing else: no markdown fences, no commentary.

Allowed vocabulary (dimension: allowed labels):
{VOCAB}

Reply format (fill in real values, keep every dimension key):
{EXAMPLE}
"""


def build_system_prompt():
    return (
        SYSTEM_TEMPLATE
        .replace("{VOCAB}", build_vocab_text())
        .replace("{EXAMPLE}", build_json_example())
    )


def build_user_prompt(heading, scene_text, max_shots):
    return (
        f"Scene heading: {heading}\n\n"
        f"Scene text:\n{scene_text}\n\n"
        f"Suggest between 2 and {max_shots} shots for this scene."
    )