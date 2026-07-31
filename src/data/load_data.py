"""
Track A (124AD0008) — load the core DTI datasets.

DAVIS:  ~27,600 drug-target pairs, 68 drugs, 379 proteins.
KIBA:   ~118,000 drug-target pairs, ~2,100 drugs, 229 proteins.

See docs/01_GUIDE_124AD0008.md Step 1 for the full explanation.
"""
from tdc.multi_pred import DTI


def load_dataset(name: str):
    """
    name: 'DAVIS' or 'KIBA'
    Returns the TDC DataLoader object (has .get_data() and .get_split()).
    """
    assert name in ("DAVIS", "KIBA"), "name must be 'DAVIS' or 'KIBA'"
    return DTI(name=name)


def summarize(data) -> None:
    """Print basic stats: useful first sanity check to share with the team."""
    df = data.get_data()
    print(f"Pairs: {len(df)}")
    print(f"Unique drugs: {df['Drug_ID'].nunique()}")
    print(f"Unique targets: {df['Target_ID'].nunique()}")
    print(f"Label range: {df['Y'].min():.3f} to {df['Y'].max():.3f}")


if __name__ == "__main__":
    for name in ("DAVIS", "KIBA"):
        print(f"\n--- {name} ---")
        data = load_dataset(name)
        summarize(data)
