# Run using: python gesture_blur_test.py 2>nul to prevent unnecessary logs
# Comprehensive test: YOLO -> Bounding Boxes -> MediaPipe Gesture Detection -> Face Blurring

import cv2
import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
import mediapipe as mp
from utils import (
    WaveDetector,
    HandOverFaceDetector,
    detect_multiple_people_yolov8,
    detect_and_blur_multiple_people,
    get_video_rotation,
    close_global_mediapipe
)

# Configuration
video_path = "C:\\Users\\Layne\\Desktop\\RECORDINGS[CONFI]\\GH010048.mp4"
GESTURE_TYPE = "wave"  # Change to "hand_over_face" to test the other detector

def detect_gesture_in_person_box_using_utils(person_box, frame_source, gesture_type="wave", fps=30, duration_seconds=2):
    """
    Use WaveDetector or HandOverFaceDetector classes from utils to detect gestures
    within a person's bounding box by analyzing the next N seconds of video.
    
    Args:
        person_box: (x1, y1, x2, y2) bounding box coordinates
        frame_source: cv2.VideoCapture object to read frames from
        gesture_type: Type of gesture to detect
        fps: Frame rate
        duration_seconds: How many seconds to analyze
    
    Returns True if gesture is detected, False otherwise.
    """
    x1, y1, x2, y2 = person_box
    frames_to_collect = int(fps * duration_seconds)  # 2 seconds worth of frames
    
    # Collect frames for the specified duration
    person_frames = []
    current_pos = frame_source.get(cv2.CAP_PROP_POS_FRAMES)  # Save current position
    
    for _ in range(frames_to_collect):
        ret, frame = frame_source.read()
        if not ret:
            break
            
        # Apply rotation if needed (assume same rotation as main video)
        # Note: You might want to pass rotation as a parameter
        
        # Extract person crop
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue
            
        person_frames.append(person_crop.copy())
    
    # Reset video position
    frame_source.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    if len(person_frames) < 10:  # Need at least 10 frames
        return False
        
    # Save the person frames as a temporary video file for the detector classes
    temp_video_path = "temp_person_crop.avi"
    
    # Create a temporary video from the person frame sequence
    height, width = person_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # Write the actual frame sequence to create a temporal video
    for frame in person_frames:
        out.write(frame)
    out.release()
    
    # Use the appropriate detector from utils
    gesture_detected = False
    try:
        if gesture_type == "wave":
            detector = WaveDetector(temp_video_path, fps, detection_confidence=0.5)
            detected_frames = detector.detect_wave_timestamps(show_ui=True, frame_skip=5)  # Show UI to see skeleton
            gesture_detected = len(detected_frames) > 0
        
        elif gesture_type == "hand_over_face":
            detector = HandOverFaceDetector(temp_video_path, fps, detection_confidence=0.3)
            detected_frames = detector.detect_hand_over_face_frames(show_ui=True, frame_skip=5)  # Show UI to see skeleton
            gesture_detected = len(detected_frames) > 0
            
    except Exception as e:
        print(f"Error in gesture detection: {e}")
        gesture_detected = False
    
    # Clean up temporary file
    try:
        import os
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    except:
        pass
    
    return gesture_detected

def main():
    # Initialize video capture and get properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation = get_video_rotation(video_path)
    cap.release()
    
    # Handle invalid FPS values
    if fps <= 0 or fps is None:
        fps = 30.0  # Default to 30 FPS if invalid
        print("Warning: Invalid FPS detected, defaulting to 30 FPS")
    
    print(f"Video Properties:")
    print(f"- FPS: {fps}")
    print(f"- Total Frames: {total_frames}")
    print(f"- Rotation: {rotation}°")
    print(f"- Testing Gesture Type: {GESTURE_TYPE}")
    print("-" * 50)
    
    # Initialize MediaPipe face detection for blurring
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # User interaction variables
    blur_enabled = False
    gesture_detected_frame = None
    
    print("\nProcessing video with YOLO -> MediaPipe gesture detection -> Face blurring...")
    print("Workflow: YOLO detects people → MediaPipe detects gestures in person boxes → Apply face blur")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0  # Safe division
    gesture_count = 0
    global_frame_skip = 120  # Process everything every 120 frames (4 seconds at 30fps)
    
    # Track which people have been processed to avoid re-processing
    processed_people = set()  # Store (frame_num, person_index) to avoid duplicates
    people_with_gestures = []  # Persistent list of people who have gestures detected
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rotation if needed
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Global processing: Only process every 30 frames (1 second at 30fps)
        if frame_count % global_frame_skip != 0:
            frame_count += 1
            continue
        
        # Step 1: YOLO person detection (runs every time we process)
        people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
        
        # Step 2: For each detected person, immediately run gesture detection for 2 seconds
        current_gesture_people = []
        for i, (x1, y1, x2, y2) in enumerate(people_detected):
            person_box = (x1, y1, x2, y2)
            
            # Create a unique identifier for this person at this frame
            person_id = (frame_count // global_frame_skip, i)  # Use processing cycle and person index
            
            # Skip if we've already processed this person recently
            if person_id in processed_people:
                continue
            
            print(f"Processing person {i+1} at frame {frame_count} - analyzing next 2 seconds...")
            
            # Run gesture detection on the next 2 seconds of video
            gesture_detected = detect_gesture_in_person_box_using_utils(
                person_box, cap, GESTURE_TYPE, fps, duration_seconds=2
            )
            
            # Mark this person as processed
            processed_people.add(person_id)
            
            if gesture_detected:
                current_gesture_people.append((x1, y1, x2, y2))
                gesture_count += 1
                
                # Ask user for blur confirmation on first gesture detection
                if not blur_enabled and gesture_detected_frame is None:
                    root = tk.Tk()
                    root.withdraw()
                    gesture_name = GESTURE_TYPE.replace("_", " ").title()
                    result = messagebox.askyesno(
                        f"{gesture_name} Detected!", 
                        f"A {gesture_name.lower()} gesture was detected in person {i+1}!\n\nDo you want to enable face blurring for people who gesture?"
                    )
                    blur_enabled = result
                    gesture_detected_frame = frame_count
                    root.destroy()
                    print(f"Gesture detected at frame {frame_count}! User chose: {'Enable' if blur_enabled else 'Disable'} blurring")
        
        # Update the persistent gesture list
        people_with_gestures = current_gesture_people
            
        # Draw bounding boxes for all detected people
        gesture_detected_people = {(x1, y1, x2, y2) for x1, y1, x2, y2 in people_with_gestures}
        
        for i, (x1, y1, x2, y2) in enumerate(people_detected):
            # Check if this person had a gesture detected
            person_box = (x1, y1, x2, y2)
            gesture_detected = person_box in gesture_detected_people
            person_id = (frame_count // global_frame_skip, i)
            
            # Draw bounding boxes - different colors for different states
            if gesture_detected:
                box_color = (0, 0, 255)  # Red for gesture detected
                label = f"Person {i+1} - {GESTURE_TYPE.upper()}!"
            elif person_id in processed_people:
                box_color = (0, 255, 0)  # Green for person already processed
                label = f"Person {i+1} - Processed"
            else:
                box_color = (0, 255, 255)  # Yellow for person waiting to be processed
                label = f"Person {i+1} - Waiting"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        # Step 3: Blur faces of people who gestured (if blurring is enabled)
        if blur_enabled and people_with_gestures:
            for x1, y1, x2, y2 in people_with_gestures:
                # Extract person region for face detection and blurring
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                
                # Detect and blur face within this person's bounding box
                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                face_result = mp_face.process(person_rgb)
                
                if face_result.detections:
                    for detection in face_result.detections:
                        box = detection.location_data.relative_bounding_box
                        # Convert relative coordinates to absolute coordinates within person box
                        fx = int(box.xmin * (x2 - x1)) + x1
                        fy = int(box.ymin * (y2 - y1)) + y1
                        fw = int(box.width * (x2 - x1))
                        fh = int(box.height * (y2 - y1))
                        
                        # Ensure coordinates are within frame bounds
                        fx, fy = max(0, fx), max(0, fy)
                        fw = min(fw, frame.shape[1] - fx)
                        fh = min(fh, frame.shape[0] - fy)
                        
                        # Apply blur to face region
                        if fw > 0 and fh > 0:
                            face_roi = frame[fy:fy+fh, fx:fx+fw]
                            blurred_face = cv2.GaussianBlur(face_roi, (55, 55), 0)
                            frame[fy:fy+fh, fx:fx+fw] = blurred_face
                            
                            # Draw face detection box (blue)
                            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 1)
                            cv2.putText(frame, "BLURRED", (fx, fy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Display frame information
        timestamp = frame_count * frame_time
        info_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Time: {timestamp:.2f}s",
            f"Step 1 - People detected: {len(people_detected)}",
            f"Step 2 - Gestures detected: {len(people_with_gestures)}",
            f"Step 3 - Blur enabled: {'YES' if blur_enabled else 'NO'}",
            f"Testing: {GESTURE_TYPE.replace('_', ' ').title()}"
        ]
        
        # Draw info with white outline for visibility
        for i, text in enumerate(info_text):
            y_pos = 30 + i*25
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add legend
        legend_y = frame.shape[0] - 100
        cv2.putText(frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "Yellow: Waiting to process", (10, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, "Green: Processed, no gesture", (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "Red: Gesture detected", (10, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, "Blue: Face blurred", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Display the frame
        cv2.imshow("YOLO → MediaPipe Gesture → Face Blur Pipeline", frame)
        
        # Check for quit or reset
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset blur state
            blur_enabled = False
            gesture_detected_frame = None
            gesture_count = 0
            print("Blur state reset!")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mp_face.close()
    close_global_mediapipe()
    
    print("\nTest completed!")
    print(f"Processed {frame_count} frames")
    print(f"Total gesture detections: {gesture_count}")
    print(f"Face blurring was {'enabled' if blur_enabled else 'disabled'}")
    if gesture_detected_frame and fps > 0:
        print(f"First gesture detected at frame {gesture_detected_frame} ({gesture_detected_frame/fps:.2f}s)")

if __name__ == "__main__":
    print("=" * 70)
    print("3-Step Pipeline: YOLO → Gesture Detection → Face Blurring")
    print("=" * 70)
    print("Workflow:")
    print("Step 1: YOLO detects person bounding boxes")
    print("Step 2: If person detected → Use gesture detector within bounding box")
    print("Step 3: If gesture detected → Blur that person's face")
    print()
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset blur state")
    print("- Color coding: Green=Person, Red=Gesture detected, Blue=Face blurred")
    print("=" * 70)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        close_global_mediapipe()
