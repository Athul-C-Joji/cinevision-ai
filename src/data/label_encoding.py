import json
# Fixed class vocabularies per dimension, built from your analysis output
#
# UPDATED: "Shot Type" split into "Shot Framing" and "Camera Angle" as two
# separate heads, to match ShotBench's 8 core dimensions (shot size, shot
# framing, camera angle, lens size, lighting type, lighting conditions,
# composition, camera movement). Camera movement is handled separately by
# the temporal model (section 2 of PROJECT_NOTES.md), so this file covers
# the remaining 7 static-frame dimensions.
#
# The original "Shot Type" tags from ShotDeck/meta.jsonl mixed framing and
# angle together (e.g. "2 shot, High angle, Overhead" describes one shot
# with both a framing property AND an angle property at once). Splitting
# just means: when encoding, run the SAME raw "Shot Type" string against
# BOTH vocabs below, and each vocab only picks up tags that belong to it.
CLASS_VOCAB = {
    "Frame Size": ['Close Up', 'Extreme Close Up', 'Extreme Wide', 'Medium',
                    'Medium Close Up', 'Medium Wide', 'Wide'],
    "Lens Size": ['Long Lens', 'Medium', 'Telephoto', 'Ultra Wide / Fisheye', 'Wide'],
    "Composition": ['Balanced', 'Center', 'Left heavy', 'Right heavy', 'Short side', 'Symmetrical'],
    "Shot Framing": ['2 shot', '3 shot', 'Clean single', 'Establishing shot',
                      'Group shot', 'Insert', 'Over the shoulder'],
    "Camera Angle": ['Aerial', 'Dutch angle', 'High angle', 'Low angle', 'Overhead'],
    "Lighting Type": ['Artificial light', 'Daylight', 'Firelight', 'Fluorescent', 'HMI',
                       'LED', 'Mixed light', 'Moonlight', 'Overcast', 'Practical light',
                       'Sunny', 'Tungsten'],
    "Lighting": ['Backlight', 'Edge light', 'Hard light', 'High contrast', 'Low contrast',
                 'Side light', 'Silhouette', 'Soft light', 'Top light', 'Underlight'],
}

# Which raw meta.jsonl field each dimension actually reads from. Needed
# because "Shot Framing" and "Camera Angle" BOTH read from the same raw
# "Shot Type" field in meta.jsonl -- there's no separate "Camera Angle"
# column in the source data, we're just splitting one field's tags into
# two output vectors.
SOURCE_FIELD = {
    "Frame Size": "Frame Size",
    "Lens Size": "Lens Size",
    "Composition": "Composition",
    "Shot Framing": "Shot Type",
    "Camera Angle": "Shot Type",
    "Lighting Type": "Lighting Type",
    "Lighting": "Lighting",
}


def encode_multihot(value: str, dimension: str) -> list:
    """Convert a raw comma-separated tag string into a multi-hot vector."""
    vocab = CLASS_VOCAB[dimension]
    vec = [0] * len(vocab)
    if value is None or (isinstance(value, float)):  # NaN check
        return vec
    tags = [t.strip() for t in value.split(",")]
    for tag in tags:
        if tag in vocab:
            idx = vocab.index(tag)
            vec[idx] = 1
    return vec


if __name__ == "__main__":
    # quick smoke test
    print(encode_multihot("Extreme Wide, Wide", "Frame Size"))
    print(encode_multihot("Soft light, Top light, Side light", "Lighting"))
    print(encode_multihot(None, "Composition"))

    # test the split: same raw "Shot Type" string, two different vocabs
    raw_shot_type = "2 shot, High angle, Overhead"
    print("Shot Framing:", encode_multihot(raw_shot_type, "Shot Framing"))
    print("Camera Angle:", encode_multihot(raw_shot_type, "Camera Angle"))