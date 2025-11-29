
import subprocess
import json
import os
from pathlib import Path
import copy

scp_script = r"C:\Users\PC\Desktop\Constellate_proteomics\SCP\SCP_KTree_JSON.py"
input_tif = r"C:\Users\PC\Desktop\Constellate_proteomics\Simulation\pulsating_dots.tif"
positive_events = r"C:\Users\PC\Desktop\Constellate_proteomics\Parameter_opt\pulsating_dots.tif_positive_timeseries.json"
negative_events = r"C:\Users\PC\Desktop\Constellate_proteomics\Parameter_opt\pulsating_dots.tif_negative_timeseries.json"

# ============================================================================
# SCORING FUNCTION
# ============================================================================

def calculate_score(true_positives, false_negatives, false_positives, true_negatives, 
                   recall_weight=0.6, precision_weight=0.4):
    """
    Calculate a composite score for parameter optimization.
    
    Parameters:
    -----------
    recall_weight : float
        Weight for recall/sensitivity (how many true positives we catch)
    precision_weight : float
        Weight for precision (how many detections are true positives)
    
    Returns:
    --------
    float : Composite score (0 to 1, higher is better)
    """
    total_positives = true_positives + false_negatives
    total_detections = true_positives + false_positives
    
    # Avoid division by zero
    if total_positives == 0:
        recall = 0
    else:
        recall = true_positives / total_positives
    
    if total_detections == 0:
        precision = 0
    else:
        precision = true_positives / total_detections
    
    # Weighted score
    score = recall_weight * recall + precision_weight * precision
    
    # Alternative: F1 score (harmonic mean)
    # if (recall + precision) > 0:
    #     score = 2 * (precision * recall) / (precision + recall)
    # else:
    #     score = 0
    
    return score

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_center_coordinates(json_path):
    """Extract x,y coordinates from center field in JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        coordinates = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'center' in item:
                    center = item['center']
                    if 'x' in center and 'y' in center:
                        coordinates.append((center['x'], center['y']))
        
        return coordinates
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

def extract_coordinates(json_path):
    """Extract x,y coordinates from detected events JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        coordinates = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if 'center' in item:
                        center = item['center']
                        if 'x' in center and 'y' in center:
                            coordinates.append((center['x'], center['y']))
                            continue
                    if 'x' in item and 'y' in item:
                        coordinates.append((item['x'], item['y']))
                    elif 'X' in item and 'Y' in item:
                        coordinates.append((item['X'], item['Y']))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    coordinates.append((item[0], item[1]))
        
        return coordinates
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

def is_within_margin(coord1, coord2, margin=3):
    """Check if two coordinates are within the specified pixel margin."""
    return abs(coord1[0] - coord2[0]) <= margin and abs(coord1[1] - coord2[1]) <= margin

def find_matches(training_spots, detected_events, margin=3):
    """Find which training spots are detected in the events."""
    matches = []
    unmatched = []
    
    for train_coord in training_spots:
        found = False
        for detect_coord in detected_events:
            if is_within_margin(train_coord, detect_coord, margin):
                matches.append({
                    'training': train_coord,
                    'detected': detect_coord,
                    'distance': ((train_coord[0] - detect_coord[0])**2 + 
                                (train_coord[1] - detect_coord[1])**2)**0.5
                })
                found = True
                break
        
        if not found:
            unmatched.append(train_coord)
    
    return matches, unmatched

# ============================================================================
# PARAMETER EVALUATION
# ============================================================================

def run_detection(params):
    """Run the SCP detection script with given parameters."""
    cmd = [
        "py", "-3.11", scp_script,
        input_tif,
        "--min_sigma", str(params['min_sigma']),
        "--max_sigma", str(params['max_sigma']),
        "--threshold", str(params['intensity_threshold']),
        "--circularity_threshold", str(params['circular_threshold']),
        "--max_displacement", str(params['max_displacement']),
        "--min_pulsation_frames", str(params['min_pulsation_frames']),
        "--intensity_variation_threshold", str(params['intensity_variation_threshold'])
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        return False, e.stderr

def find_output_json(input_tif_path, scp_script_path):
    """Find the most recent JSON output file."""
    output_dir = Path(input_tif_path).parent
    json_files = list(output_dir.glob("*.json")) + list(output_dir.glob("*.JSON"))
    
    if not json_files:
        script_dir = Path(scp_script_path).parent
        json_files = list(script_dir.glob("*.json")) + list(script_dir.glob("*.JSON"))
    
    if json_files:
        return max(json_files, key=lambda p: p.stat().st_mtime)
    return None

def evaluate_parameters(params, p_training_spots, n_training_spots, verbose=False):
    """
    Evaluate a parameter set and return performance metrics.
    
    Returns:
    --------
    dict : Contains score, TP, FN, FP, TN, recall, precision
    """
    # Run detection
    success, output = run_detection(params)
    if not success:
        return {'score': 0, 'error': 'Script execution failed'}
    
    # Find output JSON
    detected_json = find_output_json(input_tif, scp_script)
    if not detected_json:
        return {'score': 0, 'error': 'No output JSON found'}
    
    # Extract detected events
    detected_events = extract_coordinates(detected_json)
    
    # Calculate matches
    p_matches, p_unmatched = find_matches(p_training_spots, detected_events, margin=3)
    n_matches, n_unmatched = find_matches(n_training_spots, detected_events, margin=3)
    
    # Calculate metrics
    true_positives = len(p_matches)
    false_negatives = len(p_unmatched)
    false_positives = len(n_matches)
    true_negatives = len(n_unmatched)
    
    total_positives = len(p_training_spots)
    total_negatives = len(n_training_spots)
    
    recall = true_positives / total_positives if total_positives > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Calculate score
    score = calculate_score(true_positives, false_negatives, false_positives, true_negatives)
    
    results = {
        'score': score,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'recall': recall,
        'precision': precision,
        'detected_count': len(detected_events)
    }
    
    if verbose:
        print(f"  Score: {score:.4f} | Recall: {recall:.2%} | Precision: {precision:.2%} | "
              f"TP: {true_positives} | FP: {false_positives}")
    
    return results

# ============================================================================
# PARAMETER OPTIMIZATION
# ============================================================================

def optimize_parameters(initial_params, p_training_spots, n_training_spots, 
                       steps_per_param=10, step_sizes=None):
    """
    Optimize parameters by testing each parameter individually across multiple steps.
    
    Parameters:
    -----------
    initial_params : dict
        Starting parameter values
    steps_per_param : int
        Number of steps to test for each parameter (in each direction)
    step_sizes : dict
        Step size for each parameter
    """
    
    # Default step sizes
    if step_sizes is None:
        step_sizes = {
            'min_sigma': 0.1,
            'max_sigma': 0.25,
            'intensity_threshold': 0.025,
            'circular_threshold': 0.05,
            'max_displacement': 0.25,
            'min_pulsation_frames': 1,
            'intensity_variation_threshold': 0.025
        }
    
    # Parameter bounds (to prevent unreasonable values)
    param_bounds = {
        'min_sigma': (0, 2),
        'max_sigma': (0.5, 5),
        'intensity_threshold': (0.01, 0.5),
        'circular_threshold': (0.1, 1.0),
        'max_displacement': (0.5, 5),
        'min_pulsation_frames': (3, 30),
        'intensity_variation_threshold': (0.05, 0.5)
    }
    
    current_params = copy.deepcopy(initial_params)
    
    # Evaluate initial parameters
    print("="*80)
    print("PARAMETER OPTIMIZATION STARTED")
    print("="*80)
    print(f"Strategy: Test {steps_per_param} steps for each parameter individually")
    print("="*80)
    print("\nEvaluating initial parameters...")
    current_results = evaluate_parameters(current_params, p_training_spots, n_training_spots, verbose=True)
    best_score = current_results['score']
    best_params = copy.deepcopy(current_params)
    best_results = current_results
    
    history = [{
        'iteration': 0,
        'param': 'initial',
        'params': copy.deepcopy(current_params),
        'results': current_results
    }]
    
    print(f"\nInitial Score: {best_score:.4f}")
    print(f"Initial Parameters: {current_params}")
    
    iteration = 0
    
    # Optimize each parameter independently
    for param_name in current_params.keys():
        if param_name not in step_sizes:
            continue
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZING PARAMETER: {param_name}")
        print(f"{'='*80}")
        
        step = step_sizes[param_name]
        original_value = current_params[param_name]
        min_val, max_val = param_bounds.get(param_name, (float('-inf'), float('inf')))
        
        # Test values in both directions
        test_values = []
        
        # Negative direction
        for i in range(1, steps_per_param + 1):
            test_val = original_value - i * step
            if test_val >= min_val:
                test_values.append(test_val)
        
        # Positive direction
        for i in range(1, steps_per_param + 1):
            test_val = original_value + i * step
            if test_val <= max_val:
                test_values.append(test_val)
        
        # Current value
        test_values.insert(len([v for v in test_values if v < original_value]), original_value)
        
        print(f"Testing {len(test_values)} values for {param_name}")
        print(f"Range: [{min(test_values):.3f}, {max(test_values):.3f}]")
        print(f"Current value: {original_value}")
        
        param_best_score = best_score
        param_best_value = original_value
        param_best_results = None
        
        # Test each value
        for test_value in test_values:
            iteration += 1
            
            if test_value == original_value:
                print(f"\n[{iteration}] {param_name} = {test_value:.4f} (current)")
                print(f"  Score: {best_score:.4f} (already evaluated)")
                continue
            
            # Test new parameter value
            test_params = copy.deepcopy(current_params)
            test_params[param_name] = test_value
            
            print(f"\n[{iteration}] Testing {param_name} = {test_value:.4f}")
            test_results = evaluate_parameters(test_params, p_training_spots, n_training_spots, verbose=True)
            
            # Track best for this parameter
            if test_results['score'] > param_best_score:
                param_best_score = test_results['score']
                param_best_value = test_value
                param_best_results = test_results
                print(f"  ✓ NEW BEST for {param_name}! Score: {param_best_score:.4f}")
        
        # Update current parameters if improvement found
        if param_best_score > best_score:
            print(f"\n{'='*80}")
            print(f"IMPROVEMENT FOUND!")
            print(f"{'='*80}")
            print(f"Parameter: {param_name}")
            print(f"Old value: {original_value:.4f} -> New value: {param_best_value:.4f}")
            print(f"Old score: {best_score:.4f} -> New score: {param_best_score:.4f}")
            print(f"Improvement: {(param_best_score - best_score):.4f} ({((param_best_score - best_score)/best_score*100):.1f}%)")
            
            best_score = param_best_score
            best_results = param_best_results
            current_params[param_name] = param_best_value
            best_params = copy.deepcopy(current_params)
            
            history.append({
                'iteration': iteration,
                'param': param_name,
                'params': copy.deepcopy(best_params),
                'results': best_results
            })
        else:
            print(f"\nNo improvement found for {param_name}. Keeping original value: {original_value:.4f}")
    
    # Final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nTotal iterations: {iteration}")
    print(f"Best Score: {best_score:.4f}")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        orig_value = initial_params[param]
        if value != orig_value:
            print(f"  {param}: {orig_value} -> {value} ✓")
        else:
            print(f"  {param}: {value}")
    
    print(f"\nBest Performance:")
    print(f"  Recall: {best_results['recall']:.2%}")
    print(f"  Precision: {best_results['precision']:.2%}")
    print(f"  True Positives: {best_results['true_positives']}")
    print(f"  False Positives: {best_results['false_positives']}")
    print(f"  False Negatives: {best_results['false_negatives']}")
    print(f"  True Negatives: {best_results['true_negatives']}")
    
    return best_params, best_results, history

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load training data
    print("Loading training data...")
    p_training_spots = extract_center_coordinates(positive_events)
    n_training_spots = extract_center_coordinates(negative_events)
    
    print(f"Positive training spots: {len(p_training_spots)}")
    print(f"Negative training spots: {len(n_training_spots)}")
    
    if len(p_training_spots) == 0 or len(n_training_spots) == 0:
        print("ERROR: Training data is empty. Cannot optimize.")
        exit(1)
    
    # Initial parameters
    initial_params = {
        'min_sigma': 0,
        'max_sigma': 3.5,
        'intensity_threshold': 0.1,
        'circular_threshold': 0.5,
        'max_displacement': 1,
        'min_pulsation_frames': 10,
        'intensity_variation_threshold': 0.20
    }
    
    # Run optimization
    # You can customize step_sizes for finer/coarser optimization:
    # custom_step_sizes = {
    #     'intensity_threshold': 0.01,  # Finer steps
    #     'max_sigma': 0.5,             # Coarser steps
    # }
    
    best_params, best_results, history = optimize_parameters(
        initial_params, 
        p_training_spots, 
        n_training_spots,
        steps_per_param=10  # Test 10 steps in each direction for each parameter
        # step_sizes=custom_step_sizes  # Optionally override default step sizes
    )
    
    # Save results to file
    output_file = "optimization_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'best_parameters': best_params,
            'best_results': best_results,
            'optimization_history': history
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")