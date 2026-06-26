# params_loader.py
import json
import ast
import numpy as np
from pathlib import Path
from constants import pc

def load_params_from_file(filename):
    """
    Load parameters from a text file.
    
    Supports:
    - JSON format (.json)
    - Python dict format (.txt or .py)
    
    Parameters:
    -----------
    filename : str
        Path to the parameters file
    
    Returns:
    --------
    params : dict
        Dictionary with parameters
    """
    filename = Path(filename)
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Try JSON first
    if filename.suffix == '.json':
        params = json.loads(content)
    else:
        # Try Python dict
        try:
            params = ast.literal_eval(content)
        except (SyntaxError, ValueError):
            # Fallback: eval for files with comments
            lines = []
            for line in content.split('\n'):
                if '#' in line:
                    line = line[:line.index('#')]
                lines.append(line)
            content_clean = ' '.join(lines)
            content_clean = content_clean.replace(',\n', ',')
            content_clean = content_clean.replace('\n', '')
            params = eval(content_clean)
    
    # Ensure numeric types for all numeric parameters
    numeric_keys = ['Mdot', 'Vw', 'Vstar', 'n_ism', 'dist', 'inclination', 'PA', 'T_ism', 'R_str', 'f_NTp', 'f_NTe', 'f_B']
    for key in numeric_keys:
        if key in params:
            params[key] = float(params[key])
    
    return params


def get_source_params(source_name, params_dir='.'):
    """
    Load parameters for a specific source.
    
    Parameters:
    -----------
    source_name : str
        Name of the source (e.g., 'RxJ0528+2838')
    params_dir : str
        Directory containing parameter files
    
    Returns:
    --------
    params : dict
        Dictionary with parameters
    """
    # Try different extensions
    for ext in ['.txt', '.json', '.py', '.dat']:
        filename = Path(params_dir) / f"{source_name}{ext}"
        if filename.exists():
            return load_params_from_file(filename)
    
    raise FileNotFoundError(f"No parameter file found for {source_name}")


def validate_params(params):
    """
    Validate and ensure all required parameters exist with correct types.
    
    Required parameters (must be in file):
    - Mdot: float > 0
    - Vw: float > 0
    - Vstar: float > 0
    - n_ism: float > 0
    - inclination: float (0°-180°)
    - PA : float (0°-360°)
    
    Optional parameters (can be set by sliders or defaults):
    - dist: float > 0 (default: 224.0 pc)
    """
    # Required parameters (must be in file)
    required = ['Mdot', 'Vw', 'Vstar', 'n_ism', 'dist']
    
    for req in required:
        if req not in params:
            raise KeyError(f"Missing required parameter in file: {req}")
    
    # Type checking and validation for required parameters
    if params['Mdot'] <= 0:
        raise ValueError(f"Mdot must be > 0, got {params['Mdot']}")
    if params['Vw'] <= 0:
        raise ValueError(f"Vw must be > 0, got {params['Vw']}")
    if params['Vstar'] <= 0:
        raise ValueError(f"Vstar must be > 0, got {params['Vstar']}")
    if params['n_ism'] <= 0:
        raise ValueError(f"n_ism must be > 0, got {params['n_ism']}")
    if params['dist'] <= 0:
        raise ValueError(f"dist must be > 0, got {params['dist']}")
    if not (0 <= params['inclination'] <= 180):
        raise ValueError(f"inclination must be between 0 and 180, got {params['inclination']}")
    if not (0 <= params['PA'] < 360):
        raise ValueError(f"PA must be between 0 and 360, got {params['PA']}")
    
    # Set default for inclination if not present
    if 'inclination' not in params:
        params['inclination'] = 60.
        print(f"Note: 'inclination' not found in file. Using default: {params['inclination']}°")

    # Set default for PA if not present
    if 'PA' not in params:
        params['PA'] = 90.
        print(f"Note: 'PA' not found in file. Using default: {params['PA']}°")

    # Set default for R_str if not present
    if 'R_str' not in params:
        params['R_str'] = 0.1*pc
        print(f"Note: 'R_str' not found in file. Using default: {params['R_str']} cm")
    
    return params