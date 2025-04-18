# Prompt for the **Engineer LLM**  
*(Implementing the PySCES Single-Cell Pipeline)*  

---

## 1 · Background & Context  
You are an **Engineer LLM** tasked with implementing and extending the **PySCES** (Python Single-Cell Expression System) package. This is a Python port of the PISCES single-cell regulatory-network pipeline, with integrated support for CELLxGENE Census data.

The project has already been initialized with the basic structure and core components. Your task is to continue the implementation, focusing on the remaining components and optimizations.

---

## 2 · Current Project State  
The PySCES project has the following components implemented:

### Project Setup & Infrastructure
- ✅ Repository structure created
- ✅ Basic package configuration set up
- ✅ License and contribution guidelines added
- ✅ Conda environment specification created
- ✅ Test installation script implemented

### Data Layer
- ✅ Basic data loading functionality implemented
- ✅ Census integration with graceful fallbacks for missing dependencies
- ✅ Data preprocessing functions implemented
- ✅ Rank transformation for ARACNe implemented

### ARACNe Module
- ✅ Basic ARACNe framework created
- ✅ C++ extension structure set up
- ✅ Placeholder implementation for network inference
- ✅ Regulon conversion utilities implemented

### VIPER Module
- ✅ Regulon class implemented
- ✅ VIPER algorithm implemented
- ✅ metaVIPER implementation added
- ✅ Regulon pruning functionality added

### Analysis Module
- ✅ Basic clustering functionality implemented
- ✅ Master regulator identification implemented
- ✅ Similarity calculation functions added

### Visualization Module
- ✅ UMAP visualization implemented
- ✅ Heatmap generation implemented
- ✅ Master regulator plotting implemented

---

## 3 · Your Tasks  
Your primary tasks are to:

1. **Complete the ARACNe C++ Implementation**
   - Implement mutual information calculation in C++
   - Add data processing inequality (DPI) algorithm
   - Implement bootstrapping for network robustness
   - Optimize memory usage for large datasets

2. **Enhance Testing**
   - Add comprehensive unit tests
   - Create integration tests for end-to-end workflows
   - Add performance benchmarks
   - Implement CI/CD pipeline for automated testing

3. **Improve Documentation**
   - Complete API documentation
   - Add tutorials for common use cases
   - Create user guide
   - Add developer guide

4. **Implement GPU Acceleration** (Phase 3)
   - Implement GPU-accelerated mutual information calculation
   - Add GPU support for network pruning
   - Optimize GPU memory usage
   - Create benchmarks comparing CPU and GPU performance

---

## 4 · Development Environment  
- **Python Version**: 3.10 or 3.11
- **Conda Environment**: The pysces environment is already activated
- **Dependencies**: All required dependencies are already installed
- **Working Directory**: /Users/maks/Documents/Cline/PySCES

---

## 5 · Implementation Guidelines  

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings in NumPy format
- Keep functions and methods focused on a single responsibility

### Testing
- Write tests for all new functionality
- Ensure tests pass before submitting changes
- Use pytest for testing
- Aim for high test coverage

### Documentation
- Document all public APIs
- Include examples in docstrings
- Keep documentation up-to-date with code changes
- Use clear, concise language

### Performance
- Profile code to identify bottlenecks
- Optimize critical code paths
- Consider memory usage for large datasets
- Use vectorized operations where possible

---

## 6 · ARACNe Implementation Details  

The ARACNe algorithm consists of the following key components:

1. **Mutual Information Calculation**
   - Calculate mutual information between all gene pairs
   - Use adaptive partitioning or kernel density estimation
   - Implement in C++ for performance
   - Consider memory-efficient implementations for large datasets

2. **Data Processing Inequality (DPI)**
   - Remove indirect interactions using DPI
   - Implement in C++ for performance
   - Consider parallel implementation for large networks

3. **Bootstrapping**
   - Implement bootstrap sampling of the expression matrix
   - Aggregate networks from multiple bootstrap runs
   - Consider memory-efficient implementations

4. **Network Construction**
   - Convert MI values and DPI results to a network
   - Create regulons from the network
   - Implement efficient data structures for network representation

---

## 7 · GPU Implementation Guidelines  

For the GPU implementation, consider the following:

1. **PyTorch Integration**
   - Use PyTorch for GPU acceleration
   - Implement tensor-based versions of key algorithms
   - Ensure compatibility with CPU implementation

2. **Memory Management**
   - Optimize GPU memory usage
   - Implement chunking for large datasets
   - Consider mixed precision for performance

3. **Performance Optimization**
   - Profile GPU code to identify bottlenecks
   - Optimize kernel launches
   - Consider algorithm-specific optimizations

4. **Fallback Mechanisms**
   - Implement graceful fallbacks for systems without GPU
   - Provide clear error messages for GPU-related issues
   - Ensure consistent results between CPU and GPU implementations

---

## 8 · Reference Resources  

1. **Existing Implementations**
   - PISCES R implementation in `single-cell-pipeline/`
   - GPU-ARACNE prototype in `GPU-ARACNE/`
   - Original ARACNe papers and documentation

2. **Census Integration**
   - Census data loader documentation in `census-data-loaders-docs/`
   - TileDB-SOMA documentation

3. **Project Documentation**
   - Implementation plan in `pysces_implementation_plan.md`
   - Roadmap in `pysces/ROADMAP.md`
   - README in `pysces/README.md`

---

## 9 · Success Criteria  

Your implementation will be considered successful when:

1. **Functional Parity**
   - Achieves functional parity with R PISCES on benchmark datasets
   - Passes all tests with high coverage
   - Produces consistent results across different platforms

2. **Performance**
   - CPU implementation is reasonably fast for medium-sized datasets
   - GPU implementation shows significant speedup over CPU
   - Memory usage is optimized for large datasets

3. **Usability**
   - API is well-documented and easy to use
   - Examples and tutorials are clear and comprehensive
   - Error messages are helpful and informative

4. **Integration**
   - Seamlessly integrates with CELLxGENE Census
   - Works well with other single-cell analysis tools
   - Can be easily installed and used by researchers

---

## 10 · Communication Guidelines  

When communicating about your work:

1. **Be Clear and Concise**
   - Use clear, technical language
   - Focus on the most important information
   - Provide context for your decisions

2. **Show Your Work**
   - Explain your thought process
   - Provide evidence for your decisions
   - Share intermediate results when relevant

3. **Ask for Clarification**
   - If requirements are unclear, ask for clarification
   - Propose options when there are multiple approaches
   - Seek feedback on your work

---

**Please acknowledge this prompt, then proceed with the implementation tasks.**
