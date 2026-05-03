# params_loader.py
import json
import ast
import numpy as np
from pathlib import Path

def load_params_from_file(filename):
    """
    Load parameters from a text file.
    
    Supports:
    - JSON format (.json)
    - Python dict format (.txt or .py)
    
    Special handling:
    - If 'lmb' is present, automatically computes 'lam' = 10**lmb
    - If neither 'lmb' nor 'lam' is present, lam will be set later (default from slider)
    
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
    
    # Convert lmb to lam if present (for initialization, but slider will override)
    if 'lmb' in params and 'lam' not in params:
        params['lam'] = 10**params['lmb']
    elif 'lmb' in params and 'lam' in params:
        print(f"Warning: Both 'lmb' ({params['lmb']}) and 'lam' ({params['lam']}) found. Using 'lam'.")
    
    # Ensure numeric types for all numeric parameters
    numeric_keys = ['Mdot', 'Vw', 'Vstar', 'n_ism', 'dist', 'inclination', 'lam', 'lmb']
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
        Dictionary with parameters (lam may be None if not in file)
    """
    # Try different extensions
    for ext in ['.txt', '.json', '.py']:
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
    - inclination: float (0-180 deg)
    
    Optional parameters (can be set by sliders or defaults):
    - dist: float > 0 (default: 224.0 pc)
    - lam: float (will be set by slider, default: 10**(-2.5))
    - lmb: float (alternative to lam, will be converted)
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
    
    # Set default for inclination if not present
    if 'inclination' not in params:
        params['inclination'] = 60.
        print(f"Note: 'inclination' not found in file. Using default: {params['inclination']}°")
    
    # Set default initial lam (will be controlled by slider)
    if 'lam' not in params and 'lmb' not in params:
        params['lmb'] = -2.5
        params['lam'] = 10**params['lmb']
        print(f"Note: Neither 'lam' nor 'lmb' found. Using initial lmb = {params['lmb']} (lam = {params['lam']:.5f})")
    
    return params