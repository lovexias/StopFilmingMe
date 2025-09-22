# Clean gesture-based face blurring - produces clean output video
# Two-pass processing: 1) Detect gestures, 2) Create clean blurred video

import cv2
import numpy as np
import mediapipe as mp
from utilities import (
    WaveDetector,
    HandOverFaceDetector,
    detect_multiple_people_yolov8,
    get_video_rotation,
    close_global_mediapipe,
    mp_face_global,
    match_person_to_blur_list,
    adjust_bounding_box_aspect_ratio,
    detect_gesture_in_person_box,
    blur_faces_of_person,
    PersonTracker
)

# Configuration
video_path = "C:\\Users\\Layne\\Desktop\\random\\RECORDINGS[CONFI]\\GH010048COPY.mp4"
GESTURE_TYPE = "wave"  # Change to "hand_over_face" to test the other detector
OUTPUT_PATH = "clean_blurred_output.mp4"  # Clean output video file name
SHOW_UI = True  # Set to False to disable real-time UI display
UI_SCALE_FACTOR = 0.5  # Scale factor for UI display (0.5 = half size for better performance)

def draw_person_box(frame, bbox, person_id, status="Detecting", color=(0, 255, 0)):
    """Draw bounding box around detected person with status text."""
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare text
    text = f"Person {person_id}: {status}"
    
    # Calculate text size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Background rectangle for text
    text_x = x1
    text_y = y1 - 10
    if text_y < text_height:
        text_y = y1 + text_height + 10
    
    cv2.rectangle(frame, (text_x, text_y - text_height - 5), 
                  (text_x + text_width + 10, text_y + 5), color, -1)
    
    # White text
    cv2.putText(frame, text, (text_x + 5, text_y - 5), font, font_scale, (255, 255, 255), thickness)
    
    return frame

def draw_info_panel(frame, frame_count, total_frames, pass_info, detected_gestures=0):
    """Draw information panel on the frame."""
    height, width = frame.shape[:2]
    
    # Background for info panel
    panel_height = 120
    cv2.rectangle(frame, (10, 10), (400, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (400, panel_height), (255, 255, 255), 2)
    
    # Text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    
    y_pos = 30
    line_spacing = 20
    
    # Pass information
    cv2.putText(frame, f"Pass: {pass_info}", (20, y_pos), font, font_scale, color, thickness)
    y_pos += line_spacing
    
    # Frame information
    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)", 
                (20, y_pos), font, font_scale, color, thickness)
    y_pos += line_spacing
    
    # Gesture type
    cv2.putText(frame, f"Gesture: {GESTURE_TYPE.replace('_', ' ').title()}", 
                (20, y_pos), font, font_scale, color, thickness)
    y_pos += line_spacing
    
    # Detected gestures count
    cv2.putText(frame, f"Gestures Found: {detected_gestures}", 
                (20, y_pos), font, font_scale, color, thickness)
    y_pos += line_spacing
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit, 's' to skip", 
                (20, y_pos), font, font_scale, (0, 255, 255), thickness)
    
    return frame

def scale_frame_for_display(frame, scale_factor):
    """Scale frame for display while maintaining aspect ratio."""
    if scale_factor == 1.0:
        return frame
    
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    return cv2.resize(frame, (new_width, new_height))

def analyze_person_across_entire_video(video_path, person_tracker, person_id, gesture_type, fps, rotation):
    """
    Analyze a specific person across the entire video to detect gestures.
    Returns True if gesture is detected anywhere in the video for this person.
    """
    print(f"  📹 Scanning entire video for Person ID {person_id}...")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = 30  # Check every 30 frames for efficiency
    
    gesture_detected_for_person = False
    frames_checked = 0
    person_appearances = 0
    
    for frame_num in range(0, total_frames, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply rotation
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Create UI frame if enabled
        ui_frame = frame.copy() if SHOW_UI else None
        
        # Detect people in current frame
        people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
        
        if not people_detected:
            # Show scanning progress even when no people detected
            if SHOW_UI and frames_checked % 20 == 0:  # Update every 20 frames
                progress = (frame_num / total_frames) * 100
                ui_frame = draw_info_panel(ui_frame, frame_num, total_frames, 
                                         f"Analyzing Person ID {person_id} for {gesture_type}", 
                                         person_appearances)
                
                cv2.putText(ui_frame, f"Scanning... ({progress:.1f}%)", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(ui_frame, f"Person appearances so far: {person_appearances}", (50, 230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                display_frame = scale_frame_for_display(ui_frame, UI_SCALE_FACTOR)
                cv2.imshow(f"Analyzing Person {person_id}", display_frame)
                cv2.waitKey(1)
            continue
            
        # Update tracker to match people
        current_people = person_tracker.update(frame, people_detected, frame_num)
        
        # Show UI for current analysis
        if SHOW_UI:
            progress = (frame_num / total_frames) * 100
            ui_frame = draw_info_panel(ui_frame, frame_num, total_frames, 
                                     f"Analyzing Person ID {person_id} for {gesture_type}", 
                                     person_appearances)
            
            # Draw all detected people, highlight target person
            for pid, person_data in current_people.items():
                bbox = person_data['bbox']
                if pid == person_id:
                    ui_frame = draw_person_box(ui_frame, bbox, pid, "TARGET PERSON", (0, 255, 0))
                else:
                    ui_frame = draw_person_box(ui_frame, bbox, pid, "Other", (128, 128, 128))
            
            # Add progress text
            cv2.putText(ui_frame, f"Scanning: {progress:.1f}%", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(ui_frame, f"Person appearances: {person_appearances}", (50, 230), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            display_frame = scale_frame_for_display(ui_frame, UI_SCALE_FACTOR)
            cv2.imshow(f"Analyzing Person {person_id}", display_frame)
            cv2.waitKey(1)
        
        # Check if our target person is in this frame
        if person_id in current_people:
            person_appearances += 1
            person_data = current_people[person_id]
            person_bbox = person_data['bbox']
            
            print(f"    🔍 Found Person ID {person_id} at frame {frame_num}, checking for {gesture_type}...")
            
            # Show gesture analysis in UI
            if SHOW_UI:
                ui_frame = draw_person_box(ui_frame, person_bbox, person_id, "ANALYZING GESTURE...", (255, 255, 0))
                cv2.putText(ui_frame, f"Checking for {gesture_type}...", (50, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(ui_frame, "Gesture detection window will open...", (50, 290), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                display_frame = scale_frame_for_display(ui_frame, UI_SCALE_FACTOR)
                cv2.imshow(f"Analyzing Person {person_id}", display_frame)
                cv2.waitKey(1000)  # Show for 1 second before opening gesture window
            
            # Analyze gesture in this specific occurrence
            gesture_detected = detect_gesture_in_person_box(
                person_bbox, cap, gesture_type, fps, duration_seconds=3
            )
            
            if gesture_detected:
                print(f"    ✅ {gesture_type.replace('_', ' ').title()} detected for Person ID {person_id}!")
                
                # Show success in UI
                if SHOW_UI:
                    ui_frame = draw_person_box(ui_frame, person_bbox, person_id, "GESTURE DETECTED!", (0, 255, 0))
                    cv2.putText(ui_frame, f"{gesture_type.upper()} FOUND!", (50, 260), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                    display_frame = scale_frame_for_display(ui_frame, UI_SCALE_FACTOR)
                    cv2.imshow(f"Analyzing Person {person_id}", display_frame)
                    cv2.waitKey(1000)  # Show success for 1 second
                
                gesture_detected_for_person = True
                break  # Found gesture, no need to continue
            else:
                # Show no gesture found in UI
                if SHOW_UI:
                    ui_frame = draw_person_box(ui_frame, person_bbox, person_id, "No gesture here", (255, 0, 0))
                    display_frame = scale_frame_for_display(ui_frame, UI_SCALE_FACTOR)
                    cv2.imshow(f"Analyzing Person {person_id}", display_frame)
                    cv2.waitKey(50)
                
        frames_checked += 1
        
        # Progress update every 100 frames checked
        if frames_checked % 100 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"    📊 Progress: {progress:.1f}% - Person appearances: {person_appearances}")
    
    cap.release()
    if SHOW_UI:
        cv2.destroyAllWindows()  # Close analysis window
    
    if gesture_detected_for_person:
        print(f"  ✅ RESULT: Person ID {person_id} performed {gesture_type} - will be blurred")
    else:
        print(f"  ❌ RESULT: Person ID {person_id} did NOT perform {gesture_type}")
        
    print(f"  📈 Stats: Checked {frames_checked} frames, found person in {person_appearances} frames")
    
    return gesture_detected_for_person

def first_pass_detect_gestures(video_path, fps, rotation):
    """
    First pass: Identify which people should be permanently blurred throughout the video.
    Uses PersonTracker to identify unique people and analyzes each person across the entire video.
    Returns a tuple: (list of people who should be blurred, the PersonTracker instance)
    """
    print("PASS 1: Analyzing video for gesture detection...")
    print(f"🎯 Strategy: Track unique people, then scan entire video for each person's gestures")
    
    # Initialize person tracker
    person_tracker = PersonTracker(max_disappeared=30, feature_threshold=0.3, motion_threshold=200)
    
    # STEP 1: First pass to discover all unique people in the video
    print("\n📋 STEP 1: Discovering all unique people in the video...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    discovery_frame_skip = 60  # Check every 60 frames to find people
    
    # Discover all unique people
    for frame_count in range(0, total_frames, discovery_frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rotation
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Create UI frame if enabled
        ui_frame = frame.copy() if SHOW_UI else None
        
        # Detect people and update tracker
        people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
        if people_detected:
            current_people = person_tracker.update(frame, people_detected, frame_count)
            if current_people:
                progress = (frame_count / total_frames) * 100
                print(f"  📊 Discovery Progress: {progress:.1f}% - Total unique people found: {len(person_tracker.tracked_people)}")
                
                # Draw UI if enabled
                if SHOW_UI:
                    # Draw info panel
                    ui_frame = draw_info_panel(ui_frame, frame_count, total_frames, 
                                             f"1 - Discovering People ({len(person_tracker.tracked_people)} unique)", 
                                             len(person_tracker.tracked_people))
                    
                    # Draw all detected people with their IDs
                    for person_id, person_data in current_people.items():
                        bbox = person_data['bbox']
                        is_new = person_data.get('first_seen_frame', 0) == frame_count
                        status = "NEW PERSON!" if is_new else f"ID: {person_id}"
                        color = (0, 255, 0) if is_new else (255, 255, 0)  # Green for new, yellow for existing
                        ui_frame = draw_person_box(ui_frame, bbox, person_id, status, color)
                    
                    # Scale and display
                    display_frame = scale_frame_for_display(ui_frame, UI_SCALE_FACTOR)
                    cv2.imshow("Person Discovery - Pass 1", display_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(50) & 0xFF  # Slower for better visibility
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Skip ahead in discovery
                        frame_count += discovery_frame_skip * 5  # Skip 5 intervals ahead
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        else:
            # Show UI even when no people detected
            if SHOW_UI:
                progress = (frame_count / total_frames) * 100
                ui_frame = draw_info_panel(ui_frame, frame_count, total_frames, 
                                         f"1 - Discovering People ({len(person_tracker.tracked_people)} unique)", 
                                         len(person_tracker.tracked_people))
                
                # Add "No people detected" text
                cv2.putText(ui_frame, "No people detected in this frame", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                display_frame = scale_frame_for_display(ui_frame, UI_SCALE_FACTOR)
                cv2.imshow("Person Discovery - Pass 1", display_frame)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
    
    cap.release()
    if SHOW_UI:
        cv2.destroyAllWindows()  # Close discovery window
    
    discovered_people = list(person_tracker.tracked_people.keys())
    print(f"\n🎉 STEP 1 COMPLETE: Discovered {len(discovered_people)} unique people")
    
    if not discovered_people:
        print("❌ No people found in the video!")
        return []
    
    # STEP 2: Analyze each unique person across the entire video
    print(f"\n🔍 STEP 2: Analyzing each person across entire video for {GESTURE_TYPE} gestures...")
    people_to_blur_permanently = []
    
    for i, person_id in enumerate(discovered_people):
        print(f"\n👤 Analyzing Person {i+1}/{len(discovered_people)} (ID: {person_id})")
        
        # Mark this person as being scanned
        person_tracker.mark_person_scanned(person_id)
        
        # Analyze this person across the entire video
        has_gesture = analyze_person_across_entire_video(
            video_path, person_tracker, person_id, GESTURE_TYPE, fps, rotation
        )
        
        if has_gesture:
            # Mark person for blurring
            person_tracker.mark_gesture_detected(person_id)
            
            # Get the person's bbox from tracker (use the last known position)
            person_data = person_tracker.tracked_people[person_id]
            bbox = person_data['bbox']
            person_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            people_to_blur_permanently.append({
                'person_id': person_id,
                'bbox': bbox,
                'center': person_center,
                'first_detected_frame': person_data.get('first_seen_frame', 0)
            })
            
            print(f"  ✅ Person ID {person_id} will be blurred throughout the video")
        else:
            print(f"  ❌ Person ID {person_id} will NOT be blurred")
    
    print(f"\n🎉 PASS 1 COMPLETE:")
    print(f"- Discovered {len(discovered_people)} unique people")
    print(f"- Found {len(people_to_blur_permanently)} people who should be blurred")
    print(f"- Each person was analyzed across the entire video")
    
    return people_to_blur_permanently, person_tracker  # Return both the list and the tracker

def second_pass_create_clean_video(video_path, people_to_blur, person_tracker_from_pass1, fps, rotation, frame_width, frame_height):
    """
    Second pass: Create clean output video with only the necessary face blurring.
    Uses the same PersonTracker instance from Pass 1 to maintain person identity consistency.
    Shows real-time UI if enabled.
    """
    print("\nPASS 2: Creating clean blurred video...")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use the SAME PersonTracker instance from Pass 1 to maintain ID consistency
    # Reset disappeared counts to allow redetection
    for person_id in person_tracker_from_pass1.tracked_people:
        person_tracker_from_pass1.tracked_people[person_id]['disappeared'] = 0
    
    # Extract the person IDs that should be blurred
    person_ids_to_blur = {person['person_id'] for person in people_to_blur}
    print(f"Will blur Person IDs: {person_ids_to_blur}")
    print(f"PersonTracker has {len(person_tracker_from_pass1.tracked_people)} people from Pass 1")
    
    # Optimization: Run YOLO detection every N frames
    yolo_detection_interval = 5  # Run YOLO every 5 frames for better tracking accuracy
    last_detected_people_with_masks = []  # Cache of last YOLO detections with masks
    
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
        
        # Create UI frame before any modifications
        ui_frame = frame.copy() if SHOW_UI else None
        
        # Run YOLO detection only every N frames
        if frame_count % yolo_detection_interval == 0:
            yolo_detections = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
            last_detected_people_with_masks = yolo_detections  # Keep full (bbox, mask) tuples
            if frame_count % 100 == 0:  # Debug info
                print(f"  Frame {frame_count}: YOLO detected {len(last_detected_people_with_masks)} people")
        
        # Use PersonTracker to maintain identity consistency
        current_tracked_people = {}
        if last_detected_people_with_masks:
            current_tracked_people = person_tracker_from_pass1.update(frame, last_detected_people_with_masks, frame_count)
        
        # Track which people are being blurred in this frame
        blurred_people = []
        
        # Blur faces of people who should be permanently blurred (based on person ID)
        for person_id, person_data in current_tracked_people.items():
            if person_id in person_ids_to_blur:
                bbox = person_data['bbox']
                # Blur this person's face
                frame = blur_faces_of_person(frame, bbox)
                blurred_people.append(bbox)
                
                if frame_count % 50 == 0:  # Debug info every 50 frames
                    print(f"  Frame {frame_count}: Blurring Person ID {person_id}")
            else:
                if frame_count % 50 == 0:  # Debug info every 50 frames
                    print(f"  Frame {frame_count}: NOT blurring Person ID {person_id} (not in blur list: {person_ids_to_blur})")
        
        # Draw UI if enabled
        if SHOW_UI:
            # Update info panel
            ui_frame = draw_info_panel(ui_frame, frame_count, total_frames, 
                                     f"2 - Blurring Video ({len(blurred_people)} blurred)", len(people_to_blur))
            
            # Draw all tracked people with their IDs and blur status
            for person_id, person_data in current_tracked_people.items():
                bbox = person_data['bbox']
                if person_id in person_ids_to_blur:
                    ui_frame = draw_person_box(ui_frame, bbox, person_id, "BLURRED", (0, 255, 0))
                else:
                    ui_frame = draw_person_box(ui_frame, bbox, person_id, "Normal", (0, 255, 255))
            
            # Show which person IDs should be blurred
            text_y = 150
            cv2.putText(ui_frame, f"Blur IDs: {list(person_ids_to_blur)}", (50, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show the blurred result in a small overlay
            if blurred_people:
                # Create a small overlay showing the blurred frame
                overlay_size = (200, 150)
                blurred_overlay = cv2.resize(frame, overlay_size)
                
                # Position overlay in top-right corner
                overlay_x = ui_frame.shape[1] - overlay_size[0] - 20
                overlay_y = 20
                
                # Add border
                cv2.rectangle(ui_frame, (overlay_x - 2, overlay_y - 2), 
                             (overlay_x + overlay_size[0] + 2, overlay_y + overlay_size[1] + 2), 
                             (255, 255, 255), 2)
                
                # Add overlay
                ui_frame[overlay_y:overlay_y + overlay_size[1], 
                        overlay_x:overlay_x + overlay_size[0]] = blurred_overlay
                
                # Add label
                cv2.putText(ui_frame, "Output Preview", (overlay_x, overlay_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Scale and display
            display_frame = scale_frame_for_display(ui_frame, UI_SCALE_FACTOR)
            cv2.imshow("Face Blurring - Pass 2", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Skip ahead 100 frames
                frame_count += 100
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                continue
        
        # Write clean frame to output video
        video_writer.write(frame)
        
        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    video_writer.release()
    if SHOW_UI:
        cv2.destroyAllWindows()
    
    print(f"\nPASS 2 COMPLETE:")
    print(f"- Processed {frame_count} frames")
    print(f"- YOLO detection interval: {yolo_detection_interval} frames")
    print(f"- Used PersonTracker for consistent identity matching")
    print(f"- Blurred only Person IDs: {person_ids_to_blur}")
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
    print(f"Real-time UI: {'Enabled' if SHOW_UI else 'Disabled'}")
    if SHOW_UI:
        print(f"UI Scale: {UI_SCALE_FACTOR * 100:.0f}%")
        print("Controls: 'q' = quit, 's' = skip ahead")
    print("=" * 70)
    
    # Pass 1: Detect gestures and identify people to blur
    people_to_blur, person_tracker = first_pass_detect_gestures(video_path, fps, rotation)
    
    if not people_to_blur:
        print(f"\nNo {GESTURE_TYPE.replace('_', ' ').lower()} gestures detected in the video.")
        print("Creating output video without any blurring...")
    
    # Pass 2: Create clean blurred video using the same PersonTracker
    second_pass_create_clean_video(video_path, people_to_blur, person_tracker, fps, rotation, frame_width, frame_height)
    
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