import json

# Fixed class vocabularies per dimension, built from your analysis output
CLASS_VOCAB = {
    "Frame Size": ['Close Up', 'Extreme Close Up', 'Extreme Wide', 'Medium',
                    'Medium Close Up', 'Medium Wide', 'Wide'],
    "Lens Size": ['Long Lens', 'Medium', 'Telephoto', 'Ultra Wide / Fisheye', 'Wide'],
    "Composition": ['Balanced', 'Center', 'Left heavy', 'Right heavy', 'Short side', 'Symmetrical'],
    "Shot Type": ['2 shot', '3 shot', 'Aerial', 'Clean single', 'Dutch angle',
                  'Establishing shot', 'Group shot', 'High angle', 'Insert',
                  'Low angle', 'Over the shoulder', 'Overhead'],
    "Lighting Type": ['Artificial light', 'Daylight', 'Firelight', 'Fluorescent', 'HMI',
                       'LED', 'Mixed light', 'Moonlight', 'Overcast', 'Practical light',
                       'Sunny', 'Tungsten'],
    "Lighting": ['Backlight', 'Edge light', 'Hard light', 'High contrast', 'Low contrast',
                 'Side light', 'Silhouette', 'Soft light', 'Top light', 'Underlight'],
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