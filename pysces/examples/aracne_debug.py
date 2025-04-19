"""
Debug script for ARACNe implementation.

This script creates a minimal test case to debug the ARACNe implementation
and prints detailed information about the network structure.
"""

import pysces
import numpy as np
import pandas as pd
import anndata as ad
import json
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def json_serialize(obj):
    """Helper function to serialize numpy arrays for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def add_edges_to_network(network):
    """Add an 'edges' key to the network dictionary for compatibility."""
    print("Starting add_edges_to_network function...")
    
    if not isinstance(network, dict):
        print(f"ERROR: Network is not a dictionary, it's a {type(network)}")
        return network
    
    if 'regulons' not in network:
        print(f"ERROR: 'regulons' key not found in network. Keys: {list(network.keys())}")
        return network
    
    edges = []
    print(f"Processing {len(network['regulons'])} regulons...")
    
    for tf_name, regulon_data in network['regulons'].items():
        print(f"Processing TF: {tf_name}, regulon_data type: {type(regulon_data)}")
        
        if not isinstance(regulon_data, dict):
            print(f"ERROR: regulon_data for {tf_name} is not a dictionary, it's a {type(regulon_data)}")
            continue
            
        if 'targets' not in regulon_data:
            print(f"ERROR: 'targets' key not found in regulon_data for {tf_name}. Keys: {list(regulon_data.keys())}")
            continue
            
        print(f"Processing {len(regulon_data['targets'])} targets for {tf_name}...")
        
        for target, strength in regulon_data['targets'].items():
            edges.append({
                'source': tf_name,
                'target': target,
                'weight': strength
            })
    
    print(f"Created {len(edges)} edges")
    network['edges'] = edges
    return network

def main():
    print("\n=== Creating synthetic dataset ===")
    # Create minimal test data
    n_cells = 50
    n_genes = 100
    
    # Create random count data
    X = np.random.randint(0, 10, size=(n_cells, n_genes))
    
    # Create gene names with some TFs
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    tf_names = [f"TF_{i}" for i in range(10)]
    gene_names[:10] = tf_names  # First 10 genes are TFs
    
    # Create cell names
    cell_names = [f"cell_{i}" for i in range(n_cells)]
    
    # Create AnnData object
    adata = ad.AnnData(
        X=X, 
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names)
    )
    
    print(f"Dataset shape: {adata.shape}")
    
    print("\n=== Running ARACNe with Python implementation ===")
    try:
        # Create ARACNe instance with minimal settings
        aracne = pysces.ARACNe(
            bootstraps=2,  # Small number for quick testing
            p_value=0.05,  # Less stringent for testing
            dpi_tolerance=0.1,
            use_gpu=True  # Force Python implementation
        )
        
        print("Starting ARACNe run...")
        network = aracne.run(adata, tf_list=tf_names, validate=True)
        print("ARACNe run completed")
        
        # Print detailed information about the network
        print("\n=== Network Information ===")
        print(f"Network type: {type(network)}")
        
        if network is None:
            print("Network is None!")
            return
            
        if not isinstance(network, dict):
            print(f"Network is not a dictionary, it's a {type(network)}")
            return
            
        print(f"Network keys: {list(network.keys())}")
        
        # Check regulons
        if 'regulons' in network:
            print(f"Regulons type: {type(network['regulons'])}")
            
            if isinstance(network['regulons'], dict):
                print(f"Number of TFs in regulons: {len(network['regulons'])}")
                
                # Get the first TF as an example
                if network['regulons']:
                    first_tf = next(iter(network['regulons']))
                    print(f"First TF: {first_tf}")
                    print(f"First TF regulon structure: {network['regulons'][first_tf]}")
                    
                    if 'targets' in network['regulons'][first_tf]:
                        targets = network['regulons'][first_tf]['targets']
                        print(f"Number of targets for first TF: {len(targets)}")
                        if targets:
                            first_target = next(iter(targets))
                            print(f"First target: {first_target}, strength: {targets[first_target]}")
        
        # Try to add edges to the network
        print("\n=== Adding edges to network ===")
        try:
            network = add_edges_to_network(network)
            print(f"Network now has {len(network['edges'])} edges")
        except Exception as e:
            print(f"Error adding edges to network: {str(e)}")
            traceback.print_exc()
        
        # Print full network structure as JSON
        print("\n=== Full Network Structure ===")
        try:
            # Limit the size of consensus_matrix for readability
            if 'consensus_matrix' in network:
                print("Note: consensus_matrix will be truncated for readability")
                if isinstance(network['consensus_matrix'], np.ndarray):
                    shape = network['consensus_matrix'].shape
                    network['consensus_matrix'] = network['consensus_matrix'][:2, :2]
                    print(f"Original consensus_matrix shape: {shape}")
            
            network_json = json.dumps(network, indent=2, default=json_serialize)
            print(network_json[:5000] + "..." if len(network_json) > 5000 else network_json)
        except Exception as e:
            print(f"Error serializing network to JSON: {str(e)}")
            traceback.print_exc()
        
        # Convert network to regulons
        print("\n=== Converting network to regulons ===")
        try:
            regulons = pysces.aracne_to_regulons(network)
            print(f"Created {len(regulons)} regulons")
            
            if regulons:
                first_regulon = regulons[0]
                print(f"First regulon TF: {first_regulon.tf_name}")
                print(f"First regulon has {len(first_regulon.targets)} targets")
        except Exception as e:
            print(f"Error converting network to regulons: {str(e)}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error running ARACNe: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
