# Clean gesture-based face blurring - produces clean output video
# Two-pass processing: 1) Detect gestures, 2) Create clean blurred video

import cv2
import numpy as np
import mediapipe as mp
from utils import (
    WaveDetector,
    HandOverFaceDetector,
    detect_multiple_people_yolov8,
    get_video_rotation,
    close_global_mediapipe
)

# Configuration
video_path = "C:\\Users\\Layne\\Desktop\\RECORDINGS[CONFI]\\GH010048.mp4"
GESTURE_TYPE = "wave"  # Change to "hand_over_face" to test the other detector
OUTPUT_PATH = "clean_blurred_output.mp4"  # Clean output video file name

def match_person_to_blur_list(current_bbox, blur_list, tolerance=150):
    """
    Match a current person bounding box to someone in the permanent blur list.
    Uses center point distance to identify the same person across frames.
    """
    if not blur_list:
        return False
        
    current_center = ((current_bbox[0] + current_bbox[2]) // 2, 
                     (current_bbox[1] + current_bbox[3]) // 2)
    
    for blur_person in blur_list:
        blur_center = blur_person['center']
        distance = ((current_center[0] - blur_center[0])**2 + 
                   (current_center[1] - blur_center[1])**2)**0.5
        
        if distance < tolerance:
            # Update the person's current position for future matching
            blur_person['center'] = current_center
            blur_person['bbox'] = current_bbox
            return True
    
    return False

def adjust_bounding_box_aspect_ratio(x1, y1, x2, y2, frame_shape, target_aspect_ratio=0.6):
    """Adjust bounding box to have a more reasonable aspect ratio for person detection."""
    frame_height, frame_width = frame_shape[:2]
    
    # Get original dimensions
    width = x2 - x1
    height = y2 - y1
    current_aspect_ratio = width / height if height > 0 else 1.0
    
    # If aspect ratio is already reasonable, return as-is
    if 0.4 <= current_aspect_ratio <= 1.0:
        return x1, y1, x2, y2
    
    # Center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Adjust to target aspect ratio
    if current_aspect_ratio > 1.0:  # Too wide
        # Keep height, adjust width
        new_width = int(height * target_aspect_ratio)
        new_x1 = max(0, center_x - new_width // 2)
        new_x2 = min(frame_width, center_x + new_width // 2)
        new_y1, new_y2 = y1, y2
    else:  # Too narrow (very rare)
        # Keep width, adjust height
        new_height = int(width / target_aspect_ratio)
        new_y1 = max(0, center_y - new_height // 2)
        new_y2 = min(frame_height, center_y + new_height // 2)
        new_x1, new_x2 = x1, x2
    
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

def detect_gesture_in_person_box(person_box, frame_source, gesture_type="wave", fps=30, duration_seconds=2):
    """
    Detect gestures within a person's bounding box by analyzing the next N seconds of video.
    Returns True if gesture is detected, False otherwise.
    """
    x1, y1, x2, y2 = person_box
    frames_to_collect = int(fps * duration_seconds)
    
    # Collect frames for the specified duration
    person_frames = []
    current_pos = frame_source.get(cv2.CAP_PROP_POS_FRAMES)
    
    for _ in range(frames_to_collect):
        ret, frame = frame_source.read()
        if not ret:
            break
            
        # Adjust bounding box aspect ratio
        adj_x1, adj_y1, adj_x2, adj_y2 = adjust_bounding_box_aspect_ratio(x1, y1, x2, y2, frame.shape)
        
        # Extract person crop
        person_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]
        if person_crop.size == 0:
            continue
            
        # Scale up small crops for better MediaPipe processing
        if person_crop.shape[0] < 300 or person_crop.shape[1] < 200:
            scale_factor = max(300 / person_crop.shape[0], 200 / person_crop.shape[1])
            new_height = int(person_crop.shape[0] * scale_factor)
            new_width = int(person_crop.shape[1] * scale_factor)
            person_crop = cv2.resize(person_crop, (new_width, new_height))
            
        person_frames.append(person_crop.copy())
    
    # Reset video position
    frame_source.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    if len(person_frames) < 10:
        return False
        
    # Create temporary video for gesture detection
    temp_video_path = "temp_person_crop.avi"
    height, width = person_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    for frame in person_frames:
        out.write(frame)
    out.release()
    
    # Run gesture detection
    gesture_detected = False
    try:
        if gesture_type == "wave":
            detector = WaveDetector(temp_video_path, fps, detection_confidence=0.4)
            detected_frames = detector.detect_wave_timestamps(show_ui=True, frame_skip=3)  # Show UI
            gesture_detected = len(detected_frames) > 0
        
        elif gesture_type == "hand_over_face":
            detector = HandOverFaceDetector(temp_video_path, fps, detection_confidence=0.3)
            detected_frames = detector.detect_hand_over_face_frames(show_ui=True, frame_skip=3)  # Show UI
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

def first_pass_detect_gestures(video_path, fps, rotation):
    """
    First pass: Identify which people should be permanently blurred throughout the video.
    Only starts analyzing after YOLO first detects a person.
    Returns a list of people who should be blurred.
    """
    print("PASS 1: Analyzing video for gesture detection...")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    global_frame_skip = 60  # Process every 60 frames (2 seconds at 30fps)
    processed_people = set()
    people_to_blur_permanently = []
    
    # Flag to track if we've detected the first person
    first_person_detected = False
    
    while cap.isOpened():
        # Skip directly to the next frame we want to process
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
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
        
        # Always check for people to see if we should start analyzing
        people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
        
        # If we haven't detected the first person yet, keep looking
        if not first_person_detected:
            if people_detected:
                first_person_detected = True
                print(f"First person detected at frame {frame_count}. Starting gesture analysis...")
            else:
                frame_count += global_frame_skip  # Jump to next frame
                continue
        
        # Create a copy for UI display
        display_frame = frame.copy()
        
        # Draw all detected people
        for i, (x1, y1, x2, y2) in enumerate(people_detected):
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow boxes
            cv2.putText(display_frame, f"Person {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw already processed people in different color
        for blur_person in people_to_blur_permanently:
            bbox = blur_person['bbox']
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 3)  # Magenta for permanent blur
            cv2.putText(display_frame, "WILL BE BLURRED", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Add status text
        status_texts = [
            f"PASS 1: Gesture Detection Analysis",
            f"Frame: {frame_count}",
            f"First person detected: {'YES' if first_person_detected else 'NO'}",
            f"People detected: {len(people_detected)}",
            f"People to blur: {len(people_to_blur_permanently)}",
            f"Gesture type: {GESTURE_TYPE.replace('_', ' ').title()}"
        ]
        
        for i, text in enumerate(status_texts):
            y_pos = 30 + i * 25
            cv2.putText(display_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add legend
        legend_y = display_frame.shape[0] - 80
        cv2.putText(display_frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(display_frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(display_frame, "Yellow: Person detected", (10, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display_frame, "Magenta: Will be blurred", (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.putText(display_frame, "Press 'q' to quit, 's' to skip UI", (10, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow("Pass 1: Gesture Detection Analysis", display_frame)
        
        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User quit during analysis")
            cap.release()
            cv2.destroyAllWindows()
            return people_to_blur_permanently
        elif key == ord('s'):
            print("Skipping UI for faster processing...")
            cv2.destroyAllWindows()
            break
        
        # Only run gesture analysis if people are detected (we're already on analysis frames)
        if not people_detected:
            frame_count += global_frame_skip  # Jump to next frame
            continue
        
        print(f"Frame {frame_count}: Analyzing {len(people_detected)} people for gestures...")
        
        # Check each person for gestures (no more YOLO calls needed here)
        for i, (x1, y1, x2, y2) in enumerate(people_detected):
            person_box = (x1, y1, x2, y2)
            person_id = (frame_count // global_frame_skip, i)
            
            if person_id in processed_people:
                continue
            
            print(f"  Checking person {i+1} for {GESTURE_TYPE}...")
            
            # Highlight the person being analyzed
            analysis_frame = display_frame.copy()
            cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Red border for current analysis
            cv2.putText(analysis_frame, f"ANALYZING PERSON {i+1}...", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Pass 1: Gesture Detection Analysis", analysis_frame)
            cv2.waitKey(100)  # Brief pause to show which person is being analyzed
            
            # Run gesture detection (uses MediaPipe only, no YOLO)
            gesture_detected = detect_gesture_in_person_box(
                person_box, cap, GESTURE_TYPE, fps, duration_seconds=2
            )
            
            processed_people.add(person_id)
            
            if gesture_detected:
                person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                people_to_blur_permanently.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': person_center,
                    'first_detected_frame': frame_count
                })
                print(f"  ✓ {GESTURE_TYPE.replace('_', ' ').title()} detected! Person will be blurred.")
                
                # Show success feedback
                success_frame = display_frame.copy()
                cv2.rectangle(success_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Green border for success
                cv2.putText(success_frame, f"GESTURE DETECTED!", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Pass 1: Gesture Detection Analysis", success_frame)
                cv2.waitKey(1000)  # Show success for 1 second
            else:
                print(f"  ✗ No {GESTURE_TYPE.replace('_', ' ').lower()} detected.")
        
        frame_count += global_frame_skip  # Jump to next frame
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nPASS 1 COMPLETE:")
    print(f"- Processed {frame_count} frames")
    print(f"- Found {len(people_to_blur_permanently)} people who should be blurred")
    
    return people_to_blur_permanently

def second_pass_create_clean_video(video_path, people_to_blur, fps, rotation, frame_width, frame_height):
    """
    Second pass: Create clean output video with only the necessary face blurring.
    Optimized: Run YOLO every 30 frames, use last detection for intermediate frames.
    """
    print("\nPASS 2: Creating clean blurred video...")
    
    # Initialize MediaPipe face detection
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Optimization: Run YOLO every 30 frames, use last detection for others
    yolo_skip_frames = 30  # Run YOLO every 30 frames (1 second at 30fps)
    last_people_detected = []
    
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
        
        # Run YOLO every 30 frames, otherwise use last detection
        if frame_count % yolo_skip_frames == 0:
            people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
            last_people_detected = people_detected
            if frame_count % 300 == 0:  # Progress update every 10 seconds
                print(f"  YOLO detection update at frame {frame_count} - Found {len(people_detected)} people")
        else:
            # Use last YOLO detection for intermediate frames
            people_detected = last_people_detected
        
        # Blur faces of people who should be permanently blurred
        if people_to_blur and people_detected:
            for x1, y1, x2, y2 in people_detected:
                # Check if this person should be blurred
                if match_person_to_blur_list((x1, y1, x2, y2), people_to_blur):
                    # Extract person region for face detection
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                    
                    # Detect and blur face within this person's bounding box
                    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    face_result = mp_face.process(person_rgb)
                    
                    if face_result.detections:
                        for detection in face_result.detections:
                            box = detection.location_data.relative_bounding_box
                            # Convert relative coordinates to absolute coordinates
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
        
        # Write clean frame to output video
        video_writer.write(frame)
        
        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            yolo_calls = (frame_count // yolo_skip_frames) + 1
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) | YOLO calls: {yolo_calls}")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    video_writer.release()
    mp_face.close()
    
    total_yolo_calls = (frame_count // yolo_skip_frames) + 1
    print(f"\nPASS 2 COMPLETE:")
    print(f"- Processed {frame_count} frames")
    print(f"- YOLO calls: {total_yolo_calls} (vs {frame_count} without optimization)")
    print(f"- Performance improvement: {((frame_count - total_yolo_calls) / frame_count * 100):.1f}% fewer YOLO calls")
    print(f"- Clean video saved to: {OUTPUT_PATH}")

def main():
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotation = get_video_rotation(video_path)
    cap.release()
    
    # Handle invalid FPS
    if fps <= 0 or fps is None:
        fps = 30.0
        print("Warning: Invalid FPS detected, defaulting to 30 FPS")
    
    # Adjust dimensions for rotation
    if rotation in [90, 270]:
        frame_width, frame_height = frame_height, frame_width
    
    print("=" * 70)
    print("CLEAN GESTURE-BASED FACE BLURRING")
    print("=" * 70)
    print(f"Input video: {video_path}")
    print(f"Output video: {OUTPUT_PATH}")
    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Rotation: {rotation}°")
    print(f"Gesture type: {GESTURE_TYPE.replace('_', ' ').title()}")
    print("=" * 70)
    
    # Pass 1: Detect gestures and identify people to blur
    people_to_blur = first_pass_detect_gestures(video_path, fps, rotation)
    
    if not people_to_blur:
        print(f"\nNo {GESTURE_TYPE.replace('_', ' ').lower()} gestures detected in the video.")
        print("Creating output video without any blurring...")
    
    # Pass 2: Create clean blurred video
    second_pass_create_clean_video(video_path, people_to_blur, fps, rotation, frame_width, frame_height)
    
    print(f"\n🎉 COMPLETE! Clean blurred video saved as: {OUTPUT_PATH}")
    print(f"📊 Summary: {len(people_to_blur)} people permanently blurred throughout the video")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_global_mediapipe()
