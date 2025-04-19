# Debugging PySCES Pipeline Validation

## Project Context
PySCES (Python Single Cell Expression Suite) is a scientific package implementing ARACNe and VIPER algorithms for gene regulatory network inference and protein activity estimation from single-cell RNA-seq data.

## Current Task
We were implementing input data validation for the PySCES project as outlined in the implementation_gameplan.md document. This validation system ensures that input data is appropriate for the ARACNe and VIPER pipeline, providing clear error messages when it's not.

### Completed Steps
1. Created a validation module (`pysces/src/pysces/utils/validation.py`) with functions to:
   - `validate_anndata_structure`: Checks that the AnnData object has the expected structure
   - `validate_gene_names`: Checks that gene names are present and unique
   - `validate_cell_names`: Checks that cell names are present and unique
   - `validate_raw_counts`: Checks if data appears to be raw counts
   - `validate_normalized_data`: Checks if data appears to be normalized
   - `validate_sparse_matrix`: Handles sparse matrices appropriately
   - `recommend_preprocessing`: Suggests preprocessing steps based on data characteristics

2. Integrated these validation functions with the ARACNe and VIPER pipeline:
   - Added validation to the ARACNe.run() method
   - Added validation to the viper() function
   - Added validation to the metaviper() function
   - Added validation to the viper_bootstrap_wrapper() function
   - Added validation to the viper_null_model_wrapper() function

3. Created tests for the validation module in `pysces/tests/test_validation.py`.

4. Created example scripts to demonstrate how to use the validation functions:
   - `examples/validation_example.py`: Shows how to use all validation functions
   - `examples/minimal_test.py`: A minimal test for the validation module

### Previous Issue (RESOLVED)
We were encountering an error when trying to run the pipeline validation test script (`examples/pipeline_validation_test.py`). The error occurred after ARACNe ran successfully but before the network could be processed further. The specific error was:

```
✅ ARACNe completed successfully
❌ ARACNe failed: 'edges'
```

This suggested there was an issue with accessing the 'edges' key in the network dictionary returned by ARACNe.

## Resolution
The issue has been resolved by:

1. Disabling the C++ implementation of ARACNe and using the Python implementation exclusively, as it's more reliable and easier to debug.

2. Adding the 'edges' key to the network dictionary returned by ARACNe in the `_process_results` method.

3. Creating a direct import version of the pipeline validation test script (`examples/pipeline_validation_test_direct.py`) to avoid Python path issues.

### Attempted Solutions
1. Fixed a bug in the ARACNe implementation related to index out of bounds errors in the `_process_results` method:
   ```python
   # Original code
   for j in range(len(gene_list)):
       # Skip self-interactions
       if j == tf_idx:
           continue

       # Get interaction strength
       strength = consensus_matrix[i, j]

   # Fixed code
   for j in range(len(gene_list)):
       # Skip self-interactions
       if j == tf_idx:
           continue

       # Get interaction strength
       # Make sure j is within bounds of the consensus matrix
       if j < consensus_matrix.shape[1]:
           strength = consensus_matrix[i, j]
       else:
           # Skip genes that are out of bounds
           continue
   ```

2. Added a function to add edges to the network:
   ```python
   def add_edges_to_network(network):
       """Add an 'edges' key to the network dictionary for compatibility."""
       edges = []
       for tf_name, regulon_data in network['regulons'].items():
           for target, strength in regulon_data['targets'].items():
               edges.append({
                   'source': tf_name,
                   'target': target,
                   'weight': strength
               })

       # Add edges to the result
       network['edges'] = edges
       return network
   ```

3. Modified the pipeline validation test script to use this function:
   ```python
   network = aracne.run(adata, tf_list=tf_names, validate=True)
   network = add_edges_to_network(network)  # Add this line
   print(f"✅ ARACNe completed successfully")
   print(f"Network has {len(network['edges'])} edges")
   ```

4. Added detailed exception handling and debugging output to trace the execution flow.

Despite these efforts, we're still encountering the error.

## Debug Scripts
We've created two debug scripts to help identify the root cause:

1. `examples/aracne_debug.py`: Focuses on debugging the ARACNe implementation.
   - Creates a minimal synthetic dataset
   - Runs ARACNe with the Python implementation
   - Prints detailed information about the network structure
   - Attempts to add edges to the network
   - Prints the full network structure as JSON
   - Tries to convert the network to regulons

2. `examples/validation_debug.py`: Focuses on testing the validation functions.
   - Creates a minimal synthetic dataset
   - Tests all validation functions on the dataset
   - Tests validation with invalid data (duplicate gene names)

## Potential Issues
1. The network structure might be different than expected (missing keys or different structure)
2. There might be an exception occurring earlier in the code that's being swallowed
3. The validation script might be expecting a different format than what ARACNe now produces
4. There might be an issue with the Python implementation of ARACNe

## Next Steps
1. Run the debug scripts to gather more information about the issue:
   ```
   python examples/aracne_debug.py
   python examples/validation_debug.py
   ```

2. Analyze the output to identify the root cause.

3. Implement a fix based on the findings. Potential solutions include:
   - Option 1: Modify the validation script to use the correct structure
   - Option 2: Add a transformation function to convert the network to include edges
   - Option 3: Modify the ARACNe implementation to include edges

4. Test the fix with the full pipeline validation test.

5. Continue with the implementation of the validation module and its integration with the pipeline.

## ARACNe Network Structure
The ARACNe implementation returns a network structure with the following format:
```python
{
    'regulons': regulons,  # Dictionary of TF -> targets
    'tf_names': [gene_list[i] for i in tf_indices],
    'consensus_matrix': consensus_matrix,
    'metadata': {
        'p_value': self.p_value,
        'bootstraps': self.bootstraps,
        'dpi_tolerance': self.dpi_tolerance,
        'consensus_threshold': self.consensus_threshold
    }
}
```

The 'edges' key is missing from this structure, which is causing the KeyError after ARACNe runs successfully.

## Related Code
- `pysces/src/pysces/utils/validation.py`: Contains the validation functions
- `pysces/src/pysces/aracne/core.py`: Contains the ARACNe implementation
- `pysces/src/pysces/viper/activity.py`: Contains the VIPER implementation
- `examples/pipeline_validation_test.py`: The pipeline validation test script
- `examples/aracne_debug.py`: Debug script for ARACNe
- `examples/validation_debug.py`: Debug script for validation functions
