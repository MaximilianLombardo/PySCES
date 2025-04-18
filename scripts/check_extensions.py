import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_mi_calculation(aracne_ext):
    """Test mutual information calculation with various inputs."""
    test_cases = [
        (np.array([1.0, 2.0], dtype=np.float64),
         np.array([2.0, 4.0], dtype=np.float64),
         "Perfect correlation"),
        (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
         np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
         "Identity"),
        (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
         np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64),
         "Perfect anti-correlation")
    ]
    
    for x, y, case_name in test_cases:
        logger.debug("Testing %s", case_name)
        logger.debug("x: %s", x)
        logger.debug("y: %s", y)
        mi = aracne_ext.calculate_mi_ap(x, y)
        logger.debug("MI result: %f", mi)
        
        if mi == 0.0:
            logger.warning("Unexpected zero MI for %s case", case_name)

def check_extensions():
    """Check if C++ extensions are properly compiled and loadable."""
    try:
        logger.debug("Python version: %s", sys.version)
        logger.debug("Python executable: %s", sys.executable)
        
        logger.debug("Attempting to import aracne_ext")
        from pysces.aracne._cpp import aracne_ext
        
        ext_file = aracne_ext.__file__
        logger.debug("Extension file: %s", ext_file)
        
        if os.path.exists(ext_file):
            logger.debug("File exists")
            st_mode = os.stat(ext_file).st_mode
            logger.debug("File permissions: %o", st_mode)
            logger.debug("Is executable: %s", bool(st_mode & 0o111))
        else:
            logger.error("Extension file does not exist!")
            return False
        
        logger.debug("Testing extension functionality")
        test_mi_calculation(aracne_ext)
        
        return True
        
    except Exception as e:
        logger.error("Error checking extensions: %s", str(e))
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = check_extensions()
    sys.exit(0 if success else 1)
