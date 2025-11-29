import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from skimage.feature import blob_log
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd
from PIL import Image
import tifffile 
import os
from collections import defaultdict
import argparse
import json

def load_tif_video(filepath):
    """
    Load a .tif file containing video frames
    
    Parameters:
    -----------
    filepath : str
        Path to the .tif file
        
    Returns:
    --------
    frames : np.ndarray
        Video frames with shape (n_frames, height, width)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        # Method 1: Using tifffile (recommended for scientific data)
        frames = tifffile.imread(filepath)
        print(f"Loaded using tifffile: shape {frames.shape}")
        
    except ImportError:
        print("tifffile not available, trying PIL...")
        # Method 2: Using PIL (fallback)
        with Image.open(filepath) as img:
            frames = []
            try:
                while True:
                    frame = np.array(img, dtype=np.float32)
                    frames.append(frame)
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            
            frames = np.array(frames)
            print(f"Loaded using PIL: shape {frames.shape}")
    
    # Handle different dimensionalities
    if frames.ndim == 2:
        # Single grayscale frame
        frames = frames[np.newaxis, ...]
        print("Single frame detected, added frame dimension")
    elif frames.ndim == 3:
        # Could be (n_frames, height, width) or (height, width, channels)
        if frames.shape[0] < frames.shape[2]:
            frames = np.transpose(frames, (2, 0, 1))
            print("Transposed from (height, width, n_frames) to (n_frames, height, width)")
    elif frames.ndim == 4:
        # (n_frames, height, width, channels) - convert to grayscale
        print(f"Detected color video with shape {frames.shape}")
        if frames.shape[3] == 3:  # RGB
            # Convert RGB to grayscale using standard weights
            frames = np.dot(frames[..., :3], [0.299, 0.587, 0.114])
            print(f"Converted RGB to grayscale: {frames.shape}")
        elif frames.shape[3] == 4:  # RGBA
            # Convert RGBA to grayscale (ignore alpha)
            frames = np.dot(frames[..., :3], [0.299, 0.587, 0.114])
            print(f"Converted RGBA to grayscale: {frames.shape}")
        else:
            # Just take the first channel
            frames = frames[..., 0]
            print(f"Took first channel: {frames.shape}")
    
    # Ensure we have the correct shape: (n_frames, height, width)
    if frames.ndim != 3:
        raise ValueError(f"Unexpected frame dimensions after processing: {frames.shape}")
    
    print(f"Final shape: {frames.shape}")
    print(f"Intensity range: {frames.min():.2f} to {frames.max():.2f}")
    
    return frames

def save_events_to_json(results, output_filepath):
    """
    Save detected smTIRF events coordinates to a JSON file
    
    Parameters:
    -----------
    results : dict
        Results dictionary from process_video
    output_filepath : str
        Path to save the JSON file
    """
    # Prepare data for JSON export - only coordinates
    coordinates = []
    
    # Add each valid event with just its coordinates
    for event in results['valid_events']:
        coordinates.append({
            'x': float(event['mean_position'][0]),
            'y': float(event['mean_position'][1])
        })
    
    # Save to JSON file
    with open(output_filepath, 'w') as f:
        json.dump(coordinates, f, indent=2)
    
    print(f"\nEvent coordinates saved to: {output_filepath}")
    print(f"Total events exported: {len(coordinates)}")

def process_your_tif_file(filepath, detector_params=None, save_json=True):
    """Process your .tif file with the smTIRF detector"""
    
    # Load your .tif file
    print(f"Loading {filepath}...")
    frames = load_tif_video(filepath)
    
    # Set up detector parameters
    if detector_params is None:
        detector_params = {
            'min_sigma': 1.0,
            'max_sigma': 5.0,
            'threshold': 0.1,
            'circularity_threshold': 0.7,
            'max_displacement': 2.0,
            'min_pulsation_frames': 3,
            'intensity_variation_threshold': 0.3
        }
    
    # Initialize detector
    print("Initializing detector...")
    detector = smTIRFEventDetector(**detector_params)
    
    # Show sample frame
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sample_frame = frames[frames.shape[0]//2]
    plt.imshow(sample_frame, cmap='hot')
    plt.title(f'Sample Frame')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.hist(sample_frame.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.title('Intensity Distribution')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Process the video
    print(f"Processing {frames.shape[0]} frames...")
    results = detector.process_video(frames)
    
    # Save results to JSON
    if save_json:
        # Create output filename based on input filename
        base_name = os.path.splitext(filepath)[0]
        json_filepath = f"{base_name}_events.json"
        save_events_to_json(results, json_filepath)
    
    # Visualize results
    detector.visualize_results(frames, results, frame_idx=frames.shape[0]//2)
    
    return results, frames, detector

class smTIRFEventDetector:
    """
    Single Molecule TIRF Event Detection Algorithm - Optimized Version
    
    Detects circular, fixed, and pulsating fluorescent events in microscopy videos
    """
    
    def __init__(self, 
                 min_sigma: float = 1.0,
                 max_sigma: float = 5.0,
                 threshold: float = 0.1,
                 circularity_threshold: float = 0.7,
                 max_displacement: float = 2.0,
                 min_pulsation_frames: int = 3,
                 intensity_variation_threshold: float = 0.3):
        """
        Parameters:
        -----------
        min_sigma, max_sigma : float
            Range for blob detection (roughly spot radius in pixels)
        threshold : float
            Minimum intensity threshold for blob detection
        circularity_threshold : float
            Minimum circularity (0-1, where 1 is perfect circle)
        max_displacement : float
            Maximum allowed movement between frames (pixels)
        min_pulsation_frames : int
            Minimum frames showing intensity variation for pulsation
        intensity_variation_threshold : float
            Minimum coefficient of variation for pulsation detection
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.threshold = threshold
        self.circularity_threshold = circularity_threshold
        self.max_displacement = max_displacement
        self.min_pulsation_frames = min_pulsation_frames
        self.intensity_variation_threshold = intensity_variation_threshold
        
        # Initialize merge counter
        self.merge_count = 0
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing to enhance spot detection"""
        # Convert to float
        frame = frame.astype(np.float64)
        
        # Minimal preprocessing - just smooth slightly to reduce noise
        # Don't normalize globally as it destroys intensity variations
        frame_filtered = ndimage.gaussian_filter(frame, sigma=0.5)
        
        return frame_filtered
    
    def detect_spots(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect potential spots using blob detection"""
        # Normalize only for blob detection, preserving original intensities
        frame_norm = frame.copy()
        frame_norm = (frame_norm - np.min(frame_norm)) / (np.max(frame_norm) - np.min(frame_norm) + 1e-10)
        
        # Use Laplacian of Gaussian blob detection on normalized frame
        blobs = blob_log(frame_norm, 
                        min_sigma=self.min_sigma,
                        max_sigma=self.max_sigma,
                        threshold=self.threshold,
                        overlap=0.5,
                        log_scale=False)
        
        # Convert to list of (x, y, radius, intensity) using original frame for intensity
        spots = []
        for blob in blobs:
            y, x, r = blob
            # Calculate local intensity using original (non-normalized) frame
            mask = self.create_circular_mask(frame.shape, (int(x), int(y)), int(r*1.5))
            if np.any(mask):
                intensity = np.mean(frame[mask])  # Use original frame intensities
                spots.append((x, y, r, intensity))
            
        return spots
    
    def create_circular_mask(self, shape: Tuple[int, int], center: Tuple[int, int], radius: int) -> np.ndarray:
        """Create circular mask for intensity calculations"""
        Y, X = np.ogrid[:shape[0], :shape[1]]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= radius
        return mask
    
    def assess_circularity(self, frame: np.ndarray, spot: Tuple[float, float, float, float]) -> float:
        """Assess circularity of a detected spot"""
        x, y, r, _ = spot
        
        # Create region around spot
        margin = max(int(r * 2), 5)
        x_min, x_max = max(0, int(x-margin)), min(frame.shape[1], int(x+margin))
        y_min, y_max = max(0, int(y-margin)), min(frame.shape[0], int(y+margin))
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            return 0.0
        
        # Use Otsu thresholding or mean + std
        threshold_val = np.mean(roi) + 0.5 * np.std(roi)
        binary = roi > threshold_val
        
        if not np.any(binary):
            return 0.0
        
        # Find contours
        binary_uint8 = binary.astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 5:
            return 0.0
        
        # Calculate circularity: 4π*area/perimeter²
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0 or area == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return min(circularity, 1.0)  # Cap at 1.0
    
    def analyze_pulsation(self, intensity_trace: List[float]) -> Tuple[bool, float]:
        """Analyze if intensity trace shows pulsation behavior"""
        if len(intensity_trace) < self.min_pulsation_frames:
            return False, 0.0
        
        # Calculate coefficient of variation
        mean_intensity = np.mean(intensity_trace)
        std_intensity = np.std(intensity_trace)
        
        if mean_intensity == 0:
            return False, 0.0
        
        cv = std_intensity / mean_intensity
        
        return cv >= self.intensity_variation_threshold, cv
    
    def _build_tracks_optimized(self, all_spots):
        """Optimized track building with spatial indexing and better gap handling"""
        tracks = {}
        track_id = 0
        active_tracks = {}  # Maps track_id to {'position': (x, y), 'last_frame': int}
        
        # Process each frame
        for frame_idx in range(len(all_spots)):
            current_spots = all_spots[frame_idx]
            
            if not current_spots:
                # Remove tracks that have been inactive for too long (gap > 3 frames)
                inactive_tracks = [tid for tid, data in active_tracks.items() 
                                 if frame_idx - data['last_frame'] > 3]
                for tid in inactive_tracks:
                    del active_tracks[tid]
                continue
            
            # If first frame, create initial tracks
            if frame_idx == 0:
                for spot_idx, spot in enumerate(current_spots):
                    spot_key = (frame_idx, spot_idx)
                    tracks[track_id] = {
                        'spots': {
                            spot_key: {
                                'position': (spot[0], spot[1]),
                                'intensity': spot[3],
                                'circularity': spot[4]
                            }
                        }
                    }
                    active_tracks[track_id] = {
                        'position': (spot[0], spot[1]), 
                        'last_frame': frame_idx
                    }
                    track_id += 1
                continue
            
            # For subsequent frames, use spatial indexing for efficient matching
            if not active_tracks:
                # No active tracks, create new ones
                for spot_idx, spot in enumerate(current_spots):
                    spot_key = (frame_idx, spot_idx)
                    tracks[track_id] = {
                        'spots': {
                            spot_key: {
                                'position': (spot[0], spot[1]),
                                'intensity': spot[3],
                                'circularity': spot[4]
                            }
                        }
                    }
                    active_tracks[track_id] = {
                        'position': (spot[0], spot[1]), 
                        'last_frame': frame_idx
                    }
                    track_id += 1
                continue
            
            # Build KDTree for efficient spatial queries
            active_track_ids = list(active_tracks.keys())
            active_positions = [active_tracks[tid]['position'] for tid in active_track_ids]
            current_positions = [(s[0], s[1]) for s in current_spots]
            
            if active_positions and current_positions:
                # Use KDTree for efficient nearest neighbor search
                tree = cKDTree(active_positions)
                
                # Find matches within distance threshold
                used_spots = set()
                used_tracks = set()
                
                # For each current spot, find closest active track
                for spot_idx, spot_pos in enumerate(current_positions):
                    if spot_idx in used_spots:
                        continue
                    
                    # Query KDTree for nearby tracks
                    distances, indices = tree.query(spot_pos, k=min(5, len(active_positions)))
                    
                    # Handle single result case
                    if np.isscalar(distances):
                        distances = [distances]
                        indices = [indices]
                    
                    # Find best unused track within threshold
                    best_track_idx = None
                    best_distance = float('inf')
                    
                    for dist, idx in zip(distances, indices):
                        track_id_candidate = active_track_ids[idx]
                        if (track_id_candidate not in used_tracks and 
                            dist <= self.max_displacement and 
                            dist < best_distance):
                            best_track_idx = idx
                            best_distance = dist
                    
                    if best_track_idx is not None:
                        # Assign spot to track
                        track_id_match = active_track_ids[best_track_idx]
                        spot = current_spots[spot_idx]
                        spot_key = (frame_idx, spot_idx)
                        
                        tracks[track_id_match]['spots'][spot_key] = {
                            'position': (spot[0], spot[1]),
                            'intensity': spot[3],
                            'circularity': spot[4]
                        }
                        
                        # Update active track info
                        active_tracks[track_id_match] = {
                            'position': (spot[0], spot[1]),
                            'last_frame': frame_idx
                        }
                        
                        used_tracks.add(track_id_match)
                        used_spots.add(spot_idx)
                
                # Create new tracks for unmatched spots
                for spot_idx in range(len(current_spots)):
                    if spot_idx not in used_spots:
                        spot = current_spots[spot_idx]
                        spot_key = (frame_idx, spot_idx)
                        tracks[track_id] = {
                            'spots': {
                                spot_key: {
                                    'position': (spot[0], spot[1]),
                                    'intensity': spot[3],
                                    'circularity': spot[4]
                                }
                            }
                        }
                        active_tracks[track_id] = {
                            'position': (spot[0], spot[1]),
                            'last_frame': frame_idx
                        }
                        track_id += 1
            
            # Clean up inactive tracks (gap > 3 frames)
            inactive_tracks = [tid for tid, data in active_tracks.items() 
                             if frame_idx - data['last_frame'] > 3]
            for tid in inactive_tracks:
                del active_tracks[tid]
        
        return tracks
    
    def _merge_overlapping_tracks_optimized(self, tracks, position_threshold=1.5):
        """Highly optimized track merging using spatial indexing and Union-Find"""
        if len(tracks) <= 1:
            return tracks
        
        track_list = list(tracks.items())
        n_tracks = len(track_list)
        
        # Reset merge counter
        self.merge_count = 0
        
        # Calculate mean positions for all tracks (vectorized)
        track_positions = []
        track_ids = []
        
        for track_id, track in track_list:
            positions = [data['position'] for data in track['spots'].values()]
            mean_pos = np.mean(positions, axis=0)
            track_positions.append(mean_pos)
            track_ids.append(track_id)
        
        track_positions = np.array(track_positions)
        
        # Use KDTree for efficient spatial queries
        tree = cKDTree(track_positions)
        
        # Union-Find data structure for efficient merging
        parent = list(range(n_tracks))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                self.merge_count += 1
                return True
            return False
        
        # Find all pairs within threshold distance
        pairs = tree.query_pairs(position_threshold)
        
        # Union tracks that should be merged
        for i, j in pairs:
            if union(i, j):
                print(f"Merging track {track_ids[i]} into track {track_ids[j]} "
                      f"(distance: {np.linalg.norm(track_positions[i] - track_positions[j]):.2f})")
        
        # Group tracks by their root parent
        groups = defaultdict(list)
        for i in range(n_tracks):
            root = find(i)
            groups[root].append(i)
        
        # Create merged tracks
        merged_tracks = {}
        merged_track_id = 0
        
        for group_indices in groups.values():
            # Merge all spots from tracks in this group
            merged_spots = {}
            for idx in group_indices:
                track_id, track = track_list[idx]
                merged_spots.update(track['spots'])
            
            merged_tracks[merged_track_id] = {'spots': merged_spots}
            merged_track_id += 1
        
        print(f"Total track merges counted: {self.merge_count}")
        return merged_tracks
    
    def _analyze_tracks(self, tracks):
        """Analyze tracks to find valid events with optimized calculations"""
        # First merge overlapping tracks and count merges
        merged_tracks = self._merge_overlapping_tracks_optimized(tracks, position_threshold=1)
        
        print(f"After merging: {len(tracks)} tracks → {len(merged_tracks)} tracks")
        
        valid_events = []
        
        # Pre-filter tracks by minimum length
        candidate_tracks = {tid: track for tid, track in merged_tracks.items() 
                          if len(track['spots']) >= self.min_pulsation_frames}
        
        for tid, track in candidate_tracks.items():
            # Extract and sort data once
            sorted_spots = sorted(track['spots'].items())
            positions = np.array([data['position'] for _, data in sorted_spots])
            intensities = np.array([data['intensity'] for _, data in sorted_spots])
            circularities = np.array([data['circularity'] for _, data in sorted_spots])
            
            # Vectorized calculations
            # 1. Circularity (already calculated)
            mean_circularity = np.mean(circularities)
            is_circular = mean_circularity >= self.circularity_threshold
            
            # 2. Fixed position (vectorized displacement calculation)
            if len(positions) > 1:
                displacements = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                max_displacement = np.max(displacements)
                is_fixed = max_displacement <= self.max_displacement * 2
            else:
                is_fixed = True
                max_displacement = 0
            
            # 3. Pulsation (vectorized)
            mean_intensity = np.mean(intensities)
            std_intensity = np.std(intensities)
            intensity_cv = std_intensity / mean_intensity if mean_intensity > 0 else 0
            is_pulsating = intensity_cv >= self.intensity_variation_threshold
            
            # Debug output for first few tracks
            if tid < 5:
                print(f"  Track {tid}: duration={len(track['spots'])}, circular={mean_circularity:.3f}, "
                      f"fixed={is_fixed} (max_disp={max_displacement:.1f}), pulsating={is_pulsating} (cv={intensity_cv:.3f})")
            
            # Valid event must satisfy all criteria
            if is_circular and is_fixed and is_pulsating:
                valid_events.append({
                    'track_id': tid,
                    'duration': len(track['spots']),
                    'mean_position': np.mean(positions, axis=0),
                    'mean_intensity': mean_intensity,
                    'mean_circularity': mean_circularity,
                    'max_displacement': max_displacement,
                    'intensity_cv': intensity_cv
                })
        
        return valid_events
    
    def process_video(self, frames: np.ndarray) -> Dict:
        """
        Process entire video to detect and count smTIRF events
        
        Parameters:
        -----------
        frames : np.ndarray
            Video frames with shape (n_frames, height, width)
            
        Returns:
        --------
        results : dict
            Contains detected events, tracks, statistics, and merge count
        """
        n_frames, height, width = frames.shape
        
        # Reset merge counter at the start of processing
        self.merge_count = 0
        
        # Store all detected spots per frame
        all_spots = []
        
        # Process each frame
        print("Detecting spots in frames...")
        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(f"Processing frame {i}/{n_frames}")
                
            preprocessed = self.preprocess_frame(frame)
            spots = self.detect_spots(preprocessed)
            
            # Add circularity assessment
            spots_with_circularity = []
            for spot in spots:
                circularity = self.assess_circularity(preprocessed, spot)
                spots_with_circularity.append((*spot, circularity))
            
            all_spots.append(spots_with_circularity)
            
            # Debug: print number of spots found in first few frames
            if i < 5:
                print(f"  Frame {i}: found {len(spots_with_circularity)} spots")
                if spots_with_circularity:
                    print(f"    Sample spot: pos=({spots_with_circularity[0][0]:.1f}, {spots_with_circularity[0][1]:.1f}), "
                          f"r={spots_with_circularity[0][2]:.1f}, intensity={spots_with_circularity[0][3]:.3f}, "
                          f"circularity={spots_with_circularity[0][4]:.3f}")
        
        # Track spots across frames using optimized algorithm
        print("Tracking spots across frames...")
        tracks = self._build_tracks_optimized(all_spots)
        print(f"Initial tracking found {len(tracks)} tracks")
        
        # Analyze tracks to identify valid events (this calls optimized merge function)
        print("Analyzing tracks for valid events...")
        valid_events = self._analyze_tracks(tracks)
        
        # Calculate total smTIRF events: valid events + merge count
        total_smtirf_events = len(valid_events) + self.merge_count
        
        results = {
            'total_events': len(valid_events),  # Keep original count for backward compatibility
            'total_smtirf_events': total_smtirf_events,  # NEW: Total including merges
            'merge_count': self.merge_count,  # NEW: Just the merges
            'valid_events': valid_events,
            'all_tracks': tracks,
            'frames_processed': n_frames,
            'spots_per_frame': [len(spots) for spots in all_spots]
        }
        
        print(f"Analysis complete:")
        print(f"  - Valid events found: {len(valid_events)}")
        print(f"  - Track merges counted: {self.merge_count}")
        print(f"  - Total smTIRF events: {total_smtirf_events}")
        
        return results
    
    def visualize_results(self, frames: np.ndarray, results: Dict, frame_idx: int = 0):
        """Visualize detection results on a specific frame"""
        plt.figure(figsize=(15, 10))
        
        # Show original frame
        plt.subplot(2, 3, 1)
        plt.imshow(frames[frame_idx], cmap='gray')
        plt.title(f'Original Frame {frame_idx}')
        plt.colorbar()
        
        # Show preprocessed frame
        plt.subplot(2, 3, 2)
        preprocessed = self.preprocess_frame(frames[frame_idx])
        plt.imshow(preprocessed, cmap='gray')
        plt.title(f'Preprocessed Frame {frame_idx}')
        plt.colorbar()
        
        # Show frame with all detected spots
        plt.subplot(2, 3, 3)
        plt.imshow(frames[frame_idx], cmap='gray')
        
        # Find all spots in this frame from all tracks
        for track_id, track in results['all_tracks'].items():
            for (f_idx, spot_idx), spot_data in track['spots'].items():
                if f_idx == frame_idx:
                    x, y = spot_data['position']
                    circle = plt.Circle((x, y), 3, color='yellow', fill=False, linewidth=1)
                    plt.gca().add_patch(circle)
        plt.title(f'All Tracked Spots (Frame {frame_idx})')
        
        # Show frame with valid events only
        plt.subplot(2, 3, 4)
        plt.imshow(frames[frame_idx], cmap='gray')
        
        # Overlay valid events
        for event in results['valid_events']:
            track_id = event['track_id']
            # Check if this track_id exists in all_tracks
            if track_id in results['all_tracks']:
                track = results['all_tracks'][track_id]
                
                # Find spots in this frame
                for (f_idx, spot_idx), spot_data in track['spots'].items():
                    if f_idx == frame_idx:
                        x, y = spot_data['position']
                        circle = plt.Circle((x, y), 4, color='red', fill=False, linewidth=2)
                        plt.gca().add_patch(circle)
                        plt.text(x+5, y, f'E{track_id}', color='red', fontsize=10, weight='bold')
        
        plt.title(f'Valid Events Only (Frame {frame_idx})')
        
        # Show statistics
        plt.subplot(2, 3, 5)
        if results['valid_events']:
            event_durations = [e['duration'] for e in results['valid_events']]
            plt.hist(event_durations, bins=max(10, len(event_durations)//2), alpha=0.7)
            plt.xlabel('Event Duration (frames)')
            plt.ylabel('Count')
            plt.title('Event Duration Distribution')
        else:
            plt.text(0.5, 0.5, 'No valid events found', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Event Duration Distribution')
        
        plt.subplot(2, 3, 6)
        spots_per_frame = results['spots_per_frame']
        plt.plot(spots_per_frame, 'b-', linewidth=2, label='All spots')
        plt.xlabel('Frame')
        plt.ylabel('Total Spots Detected')
        plt.title('Spots per Frame')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics with new merge information
        print(f"\nDetection Summary:")
        print(f"Total frames processed: {results['frames_processed']}")
        print(f"Total tracks found: {len(results['all_tracks'])}")
        print(f"Valid events found: {results['total_events']}")
        print(f"Track merges counted: {results['merge_count']}")
        print(f"Total smTIRF events: {results['total_smtirf_events']}")
        if results['valid_events']:
            print(f"Average event duration: {np.mean([e['duration'] for e in results['valid_events']]):.1f} frames")
            print(f"Average event intensity CV: {np.mean([e['intensity_cv'] for e in results['valid_events']]):.3f}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='smTIRF Event Detector - Analyze fluorescent microscopy videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required argument
    parser.add_argument('filepath', type=str,
                       help='Path to the .tif video file')
    
    # Optional output path for JSON
    parser.add_argument('--output_json', type=str, default=None,
                       help='Path to save JSON output (default: input_filename_events.json)')
    parser.add_argument('--no_json', action='store_true',
                       help='Disable JSON output')
    
    # Detector parameters
    parser.add_argument('--min_sigma', type=float, default=30.0,
                       help='Minimum sigma for blob detection (spot radius in pixels)')
    parser.add_argument('--max_sigma', type=float, default=40.0,
                       help='Maximum sigma for blob detection (spot radius in pixels)')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Minimum intensity threshold for blob detection')
    parser.add_argument('--circularity_threshold', type=float, default=0.5,
                       help='Minimum circularity (0-1, where 1 is perfect circle)')
    parser.add_argument('--max_displacement', type=float, default=3.0,
                       help='Maximum allowed movement between frames (pixels)')
    parser.add_argument('--min_pulsation_frames', type=int, default=5,
                       help='Minimum frames showing intensity variation for pulsation')
    parser.add_argument('--intensity_variation_threshold', type=float, default=0.2,
                       help='Minimum coefficient of variation for pulsation detection')
    
    return parser.parse_args()

def run_analysis_on_tif(filepath, params, output_json=None, save_json=True):
    """Main function to run analysis on your .tif file"""
    
    try:
        # Load frames
        print(f"Loading {filepath}...")
        frames = load_tif_video(filepath)
        
        # Initialize detector
        print("Initializing detector...")
        detector = smTIRFEventDetector(**params)
        
        # Process the video
        print(f"Processing {frames.shape[0]} frames...")
        results = detector.process_video(frames)
        
        # Save results to JSON
        if save_json:
            if output_json is None:
                # Create output filename based on input filename
                base_name = os.path.splitext(filepath)[0]
                output_json = f"{base_name}_events.json"
            save_events_to_json(results, output_json)
        
        # Visualize results
        detector.visualize_results(frames, results, frame_idx=frames.shape[0]//2)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"File: {filepath}")
        print(f"Valid events found: {results['total_events']}")
        print(f"Total smTIRF events: {results['total_smtirf_events']}")
        
        return results, frames, detector
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filepath}'")
        print("Please provide a valid file path")
        return None, None, None
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create parameter dictionary from arguments
    params = {
        'min_sigma': args.min_sigma,
        'max_sigma': args.max_sigma,
        'threshold': args.threshold,
        'circularity_threshold': args.circularity_threshold,
        'max_displacement': args.max_displacement,
        'min_pulsation_frames': args.min_pulsation_frames,
        'intensity_variation_threshold': args.intensity_variation_threshold
    }
    
    # Print parameters being used
    print("Running analysis with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # Run the analysis
    results, frames, detector = run_analysis_on_tif(
        args.filepath, 
        params, 
        output_json=args.output_json,
        save_json=not args.no_json
    )