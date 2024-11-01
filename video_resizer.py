import cv2

def resize_video(input_path, output_path, scale=None, new_width=None, new_height=None):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions
    if scale:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    
    # Set up the video writer with the new size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi format
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Write the resized frame to output
        out.write(resized_frame)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Resized video saved to {output_path}")

# Example usage
resize_video('Teste_Video.mp4', 'Teste_Video_resized.mp4', new_width=640, new_height=480)
