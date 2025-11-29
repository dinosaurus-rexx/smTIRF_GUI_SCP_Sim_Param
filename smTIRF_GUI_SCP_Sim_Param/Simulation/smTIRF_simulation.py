import cv2
import numpy as np
import random
import math
import json

# Video parameters
WIDTH, HEIGHT = 512, 512
FRAMES = 500
FPS = 30
OUTPUT_FILE = "pulsating_dots.tif"
DOT_LIST_FILE = "dot_list.json"

# Dot parameters
NUM_DOTS = 200
MOVING_PERCENTAGE = 0.1
STATIC_BRIGHT_PERCENTAGE = 0.05 
MAX_DOT_SIZE = 5
MIN_DOT_SIZE = 3
MOVE_SPEED = 2

# Filter parameters
BLUR_KERNEL_SIZE = 15  # Must be odd (3, 5, 7, 9)
BLUR_SIGMA = 5.0       

class Dot:
    def __init__(self, x, y, is_moving=False, is_static_bright=False):
        self.x = x
        self.y = y
        self.initial_x = x  # Store initial position
        self.initial_y = y
        self.is_moving = is_moving
        self.is_static_bright = is_static_bright
        self.phase = random.uniform(0, 2 * math.pi)  # Random phase for pulsation
        
        # Movement parameters for moving dots
        if is_moving:
            self.dx = random.uniform(-MOVE_SPEED, MOVE_SPEED)
            self.dy = random.uniform(-MOVE_SPEED, MOVE_SPEED)
        else:
            self.dx = 0
            self.dy = 0
    
    def update(self, frame_num):
        # Update position for moving dots
        if self.is_moving:
            self.x += self.dx
            self.y += self.dy
            
            # Bounce off edges
            if self.x <= 0 or self.x >= WIDTH:
                self.dx *= -1
                self.x = max(0, min(WIDTH, self.x))
            if self.y <= 0 or self.y >= HEIGHT:
                self.dy *= -1
                self.y = max(0, min(HEIGHT, self.y))
    
    def get_size(self, frame_num):
        # Static bright dots stay at 2x the maximum size
        if self.is_static_bright:
            return MAX_DOT_SIZE * 2
        
        # Pulsate based on frame number and phase
        # Current rate: (0.2 * 30 * 60) / (2 * 3.14159) ≈ 57.3 pulsations per minute
        pulse = math.sin((frame_num * 0.2) + self.phase)
        # Map pulse from [-1, 1] to [0, MAX_DOT_SIZE] so dots disappear at minimum
        size = MAX_DOT_SIZE * (pulse + 1) / 2
        return int(size)
    
    def is_true_event(self):
        """
        Returns True if the dot is a true event (non-moving and pulsating).
        Returns False if the dot is moving or static bright (non-pulsating).
        """
        return not self.is_moving and not self.is_static_bright
    
    def to_dict(self):
        """Convert dot to dictionary for JSON export"""
        return {
            "initial_x": self.initial_x,
            "initial_y": self.initial_y,
            "is_true_event": self.is_true_event(),
            "is_moving": self.is_moving,
            "is_static_bright": self.is_static_bright,
            "is_pulsating": not self.is_static_bright  # All non-static-bright dots pulsate
        }

def create_dots():
    """Create dots with random positions"""
    dots = []
    num_moving = int(NUM_DOTS * MOVING_PERCENTAGE)
    num_static_bright = int(NUM_DOTS * STATIC_BRIGHT_PERCENTAGE)
    num_pulsating = NUM_DOTS - num_moving - num_static_bright
    
    # Create moving dots (these will also pulsate)
    for i in range(num_moving):
        x = random.randint(50, WIDTH - 50)  # Keep some margin from edges
        y = random.randint(50, HEIGHT - 50)
        dots.append(Dot(x, y, is_moving=True, is_static_bright=False))
    
    # Create static bright dots (non-pulsating, 2x size)
    for i in range(num_static_bright):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        dots.append(Dot(x, y, is_moving=False, is_static_bright=True))
    
    # Create remaining pulsating stationary dots
    for i in range(num_pulsating):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        dots.append(Dot(x, y, is_moving=False, is_static_bright=False))
    
    return dots

def draw_frame(dots, frame_num):
    """Draw a single frame with all dots and apply soft filter"""
    # Create black background
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    for dot in dots:
        dot.update(frame_num)
        size = dot.get_size(frame_num)
        
        # Only draw circle if size > 0 (so dots can fully disappear)
        if size > 0:
            cv2.circle(frame, (int(dot.x), int(dot.y)), size, (255, 255, 255), -1)
    
    # Apply Gaussian blur to soften the circles
    blurred_frame = cv2.GaussianBlur(frame, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)
    
    return blurred_frame

def save_dot_list(dots):
    """Save dot list to JSON file"""
    dot_list = [dot.to_dict() for dot in dots]
    
    with open(DOT_LIST_FILE, 'w') as f:
        json.dump(dot_list, f, indent=2)
    
    # Print summary
    true_events = sum(1 for dot in dots if dot.is_true_event())
    false_events = len(dots) - true_events
    
    print(f"\nDot list saved to {DOT_LIST_FILE}")
    print(f"Summary:")
    print(f"  - True events (stationary & pulsating): {true_events}")
    print(f"  - False events (moving or static bright): {false_events}")
    print(f"  - Total dots: {len(dots)}")

def main():
    # Create dots
    print(f"Creating {NUM_DOTS} dots:")
    print(f"  - {int(NUM_DOTS * MOVING_PERCENTAGE)} moving & pulsating")
    print(f"  - {int(NUM_DOTS * STATIC_BRIGHT_PERCENTAGE)} static bright (2x size)")
    print(f"  - {NUM_DOTS - int(NUM_DOTS * MOVING_PERCENTAGE) - int(NUM_DOTS * STATIC_BRIGHT_PERCENTAGE)} stationary & pulsating")
    print(f"Applying Gaussian blur with kernel size {BLUR_KERNEL_SIZE} and sigma {BLUR_SIGMA}")
    dots = create_dots()
    
    # Save dot list to file
    save_dot_list(dots)
    
    print(f"\nGenerating {FRAMES} frames...")
    
    # Generate frames and store them
    frames = []
    for frame_num in range(FRAMES):
        if frame_num % 50 == 0:
            print(f"Frame {frame_num}/{FRAMES}")
        
        frame = draw_frame(dots, frame_num)
        frames.append(frame)
    
    # Save as multi-frame TIFF
    print(f"\nSaving multi-frame TIFF...")
    cv2.imwritemulti(OUTPUT_FILE, frames)
    
    print(f"\nTIFF file saved as {OUTPUT_FILE}")
    print(f"Image specs: {WIDTH}x{HEIGHT}, {FRAMES} frames, {FPS} FPS (metadata)")
    print(f"Note: TIFF doesn't store FPS natively, but frames are ordered sequentially")

if __name__ == "__main__":
    main()