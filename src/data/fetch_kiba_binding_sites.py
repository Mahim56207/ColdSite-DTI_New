import json
import urllib.request
from src.data.load_data import load_deepdta_dataset

def get_kiba_binding_sites(uniprot_id):
    """Fetches biological binding and active sites for standard UniProt IDs."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception:
        return []

    sites = []
    if 'features' in data:
        for feature in data['features']:
            # Including 'Nucleotide binding' for ATP pockets
            if feature.get('type') in ['Binding site', 'Active site', 'Site', 'Nucleotide binding']:
                loc = feature.get('location', {})
                start = loc.get('start', {}).get('value')
                end = loc.get('end', {}).get('value')
                desc = feature.get('description', 'Unknown')
                
                if start and end:
                    sites.append({
                        'start': start, 
                        'end': end, 
                        'description': desc
                    })
    return sites

def main():
    print("=== Fetching Ground Truth Binding Sites for KIBA Targets ===")
    
    # Load KIBA targets
    df = load_deepdta_dataset("kiba")
    unique_targets = df["Target_ID"].unique()
    print(f"[*] Found {len(unique_targets)} unique targets in the KIBA dataset.")
    
    ground_truth = {}
    for i, target_id in enumerate(unique_targets):
        print(f"[{i+1}/{len(unique_targets)}] Fetching sites for {target_id}...")
        sites = get_kiba_binding_sites(target_id)
        if sites:
            ground_truth[target_id] = sites
        else:
            print(f"[-] No structural binding sites found for: {target_id}")
            
    output_path = "data/kiba_ground_truth_sites.json"
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"[+] Success! Saved ground-truth binding sites for {len(ground_truth)} targets to {output_path}")

if __name__ == "__main__":
    main()