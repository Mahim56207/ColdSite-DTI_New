"""
Track A (124AD0008) — collect real binding-site ground truth from UniProt.

This is the ground truth 124AD0067's evaluation code checks the model's
attention against. See docs/01_GUIDE_124AD0008.md Step 4.

NOTE: DAVIS/KIBA protein IDs are usually gene/kinase names, not UniProt
accessions. Build (or find) a mapping table before running this at scale --
budget real time for this step, it is fiddly but important.
"""
import json
import time
import requests

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{}.json"


def get_binding_sites(uniprot_id: str) -> list:
    """
    Returns a list of {'type': 'Binding site'|'Active site', 'start': int, 'end': int}
    for the given UniProt accession ID (e.g. 'P00533').
    """
    r = requests.get(UNIPROT_URL.format(uniprot_id), timeout=15)
    r.raise_for_status()
    data = r.json()

    sites = []
    for feature in data.get("features", []):
        if feature.get("type") in ("Binding site", "Active site"):
            loc = feature["location"]
            sites.append({
                "type": feature["type"],
                "start": loc["start"]["value"],
                "end": loc["end"]["value"],
            })
    return sites


def build_ground_truth_file(uniprot_ids: list, out_path: str = "data/splits/binding_sites.json"):
    """
    Loops through a list of UniProt IDs, fetches binding sites for each,
    and saves everything to one JSON file that 124AD0067 can load directly.
    Be polite to the API: a short delay between calls avoids rate-limit issues.
    """
    ground_truth = {}
    for i, uid in enumerate(uniprot_ids):
        try:
            ground_truth[uid] = get_binding_sites(uid)
        except Exception as e:
            print(f"  [!] Failed for {uid}: {e}")
            ground_truth[uid] = []
        if i % 20 == 0:
            print(f"  {i}/{len(uniprot_ids)} done...")
        time.sleep(0.3)

    with open(out_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved binding-site ground truth for {len(uniprot_ids)} proteins to {out_path}")


if __name__ == "__main__":
    # Example only — replace with your real DAVIS/KIBA -> UniProt ID mapping
    example_ids = ["P00533", "P04626"]  # EGFR, HER2 -- just to test the function works
    for uid in example_ids:
        print(uid, get_binding_sites(uid))
