from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from PIL import Image
import io
import json
import base64
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Store uploaded TIFF data in memory (in production, use proper session management)
tiff_data = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_tiff', methods=['POST'])
def upload_tiff():
    """Handle TIFF file upload and extract all frames"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Load TIFF file
        img = Image.open(file.stream)
        frames = []
        frame_count = 0
        
        # Extract all frames
        try:
            while True:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    frame = img.convert('RGB')
                else:
                    frame = img.copy()
                
                # Convert to numpy array
                frame_array = np.array(frame)
                frames.append(frame_array)
                frame_count += 1
                
                # Move to next frame
                img.seek(img.tell() + 1)
        except EOFError:
            pass  # End of frames
        
        # Store in memory (use session ID in production)
        session_id = 'default'  # In production, generate unique session IDs
        tiff_data[session_id] = {
            'frames': frames,
            'filename': file.filename,
            'width': frames[0].shape[1],
            'height': frames[0].shape[0],
            'total_frames': len(frames)
        }
        
        return jsonify({
            'success': True,
            'total_frames': len(frames),
            'width': frames[0].shape[1],
            'height': frames[0].shape[0],
            'filename': file.filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_frame/<int:frame_num>')
def get_frame(frame_num):
    """Get a specific frame as base64 encoded image"""
    session_id = 'default'
    
    if session_id not in tiff_data:
        return jsonify({'error': 'No TIFF loaded'}), 400
    
    data = tiff_data[session_id]
    if frame_num < 0 or frame_num >= len(data['frames']):
        return jsonify({'error': 'Invalid frame number'}), 400
    
    frame = data['frames'][frame_num]
    
    # Convert numpy array to image
    img = Image.fromarray(frame.astype('uint8'))
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return jsonify({
        'image': f'data:image/png;base64,{img_str}'
    })

@app.route('/calculate_intensity', methods=['POST'])
def calculate_intensity():
    """Calculate intensity time series for a circular region"""
    session_id = 'default'
    
    if session_id not in tiff_data:
        return jsonify({'error': 'No TIFF loaded'}), 400
    
    data = tiff_data[session_id]
    params = request.json
    
    center_x = params['center_x']
    center_y = params['center_y']
    radius = 5  # Fixed 5-pixel radius
    click_frame = params['click_frame']
    fps = params['fps']
    
    # Calculate frame range (20 seconds before and after)
    start_frame = max(0, click_frame - int(20 * fps))
    end_frame = min(len(data['frames']) - 1, click_frame + int(20 * fps))
    
    intensity_data = []
    
    for frame_idx in range(start_frame, end_frame + 1):
        frame = data['frames'][frame_idx]
        
        # Extract pixels in circular region
        intensities = []
        for y in range(max(0, int(center_y - radius)), min(frame.shape[0], int(center_y + radius + 1))):
            for x in range(max(0, int(center_x - radius)), min(frame.shape[1], int(center_x + radius + 1))):
                dx = x - center_x
                dy = y - center_y
                if dx*dx + dy*dy <= radius*radius:
                    # Calculate intensity (average of RGB)
                    pixel = frame[y, x]
                    intensity = float(np.mean(pixel))
                    intensities.append(intensity)
        
        avg_intensity = np.mean(intensities) if intensities else 0
        
        intensity_data.append({
            'frame': frame_idx,
            'time': (frame_idx - click_frame) / fps,
            'intensity': round(float(avg_intensity), 2)
        })
    
    return jsonify(intensity_data)

@app.route('/export_data', methods=['POST'])
def export_data():
    """Export all region data with time series"""
    session_id = 'default'
    
    if session_id not in tiff_data:
        return jsonify({'error': 'No TIFF loaded'}), 400
    
    data = tiff_data[session_id]
    params = request.json
    
    regions = params['regions']
    fps = params['fps']
    radius = params['radius']
    
    export_data = {
        'videoInfo': {
            'name': data['filename'],
            'totalFrames': data['total_frames'],
            'width': data['width'],
            'height': data['height'],
            'fps': fps,
            'maskRadius': radius
        },
        'positive': [],
        'negative': []
    }
    
    # Process each region
    for region_type in ['positive', 'negative']:
        for region in regions.get(region_type, []):
            center_x = region['center']['x']
            center_y = region['center']['y']
            
            time_series = []
            
            # Extract data from all frames
            for frame_idx in range(data['total_frames']):
                frame = data['frames'][frame_idx]
                
                # Extract pixels in circular region
                pixels = []
                for y in range(max(0, int(center_y - radius)), min(frame.shape[0], int(center_y + radius + 1))):
                    for x in range(max(0, int(center_x - radius)), min(frame.shape[1], int(center_x + radius + 1))):
                        dx = x - center_x
                        dy = y - center_y
                        if dx*dx + dy*dy <= radius*radius:
                            pixel = frame[y, x]
                            pixels.append({
                                'x': int(x),
                                'y': int(y),
                                'r': int(pixel[0]),
                                'g': int(pixel[1]),
                                'b': int(pixel[2]),
                                'intensity': float(np.mean(pixel))
                            })
                
                time_series.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'pixelData': pixels
                })
            
            export_data[region_type].append({
                'regionId': region['id'],
                'createdAtFrame': region['createdAtFrame'],
                'center': region['center'],
                'radius': radius,
                'timeSeries': time_series
            })
    
    return jsonify(export_data)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    Path('templates').mkdir(exist_ok=True)
    app.run(debug=True, port=5000)