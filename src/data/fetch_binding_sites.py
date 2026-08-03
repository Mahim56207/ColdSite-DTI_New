import json
import time
import re
import urllib.request
import urllib.parse
from urllib.error import HTTPError

from src.data.load_data import load_deepdta_dataset

def get_binding_sites(target_id):
    """
    Fetches biological binding and active sites.
    Smartly handles both UniProt Accessions and Gene Names.
    """
    # 1. Clean the target ID robustly (e.g. "ABL1(F317I)p" -> "ABL1")
    clean_id = str(target_id).split('(')[0].strip()
    
    # 2. Try direct UniProt Accession lookup first
    url = f"https://rest.uniprot.org/uniprotkb/{clean_id}.json"
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except HTTPError as e:
        # 3. If direct lookup fails, search for the REVIEWED human gene
        if e.code in [400, 404]:
            search_query = urllib.parse.quote(f"(gene:{clean_id}) AND (organism_id:9606) AND (reviewed:true)")
            search_url = f"https://rest.uniprot.org/uniprotkb/search?query={search_query}&size=1"
            try:
                req = urllib.request.Request(search_url)
                with urllib.request.urlopen(req) as response:
                    search_data = json.loads(response.read().decode())
                    if not search_data.get('results'):
                        return []
                    data = search_data['results'][0]
            except Exception:
                return []
        else:
            return []
    except Exception:
        return []

    # 4. Extract the exact amino acid indices
    sites = []
    if 'features' in data:
        for feature in data['features']:
            # ADDED 'Nucleotide binding' which is crucial for Kinase ATP pockets!
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


if __name__ == "__main__":
    print("=== Fetching Ground Truth Binding Sites for DAVIS 442 Targets ===")
    
    # 1. Automatically load your local dataset
    df = load_deepdta_dataset("davis")
    unique_targets = df["Target_ID"].unique()
    
    print(f"[*] Found {len(unique_targets)} unique targets in the DAVIS dataset.")
    
    ground_truth = {}
    for i, target_id in enumerate(unique_targets):
        # The end='\r' rewrites the same line so your terminal doesn't get flooded
        print(f"[{i+1}/{len(unique_targets)}] Fetching sites for {target_id}...", end='\r')
        
        sites = get_binding_sites(target_id)
        if sites:
            ground_truth[target_id] = sites
        else:
            print(f"\n[-] No structural binding sites found in UniProt for: {target_id}")
            
        time.sleep(0.1)  # Gentle delay so UniProt doesn't block us
        
    out_path = "data/davis_ground_truth_sites.json"
    with open(out_path, "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"\n[+] Success! Saved ground-truth binding sites for {len(ground_truth)} targets to {out_path}")