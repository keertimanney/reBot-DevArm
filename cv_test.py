import cv2
import numpy as np

def main():
    # Initialize the camera feed (0 is usually the built-in webcam, change to 1 or 2 for external)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance if needed (optional)
        frame = cv2.resize(frame, (640, 480))
        
        # --- PREPROCESSING ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Thresholding: Invert so black objects become white (255) and light background becomes black (0)
        # Using Otsu's method to automatically find the best threshold value
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Create copies of the original frame for our 3 visualization windows
        frame_solidity = frame.copy()
        frame_hierarchy = frame.copy()
        frame_morphology = frame.copy()

        # Minimum area to filter out tiny noise speckles
        MIN_AREA = 1000 

        # =====================================================================
        # TECHNIQUE 1: Bounding Box Solidity
        # Target: Objects with low solidity (lots of empty space)
        # =====================================================================
        contours_sol, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours_sol:
            area = cv2.contourArea(c)
            if area > MIN_AREA:
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    
                    # Grippers are solid (~1.0). Lattice blocks are mostly empty (<0.6)
                    if solidity < 0.65: 
                        draw_detection(frame_solidity, c, "Solidity", (0, 255, 0))


        # =====================================================================
        # TECHNIQUE 2: Contour Hierarchy (The "Hole" Method)
        # Target: Outer boundaries that contain many inner children (holes)
        # =====================================================================
        contours_hier, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None:
            for i, c in enumerate(contours_hier):
                area = cv2.contourArea(c)
                
                # Check if it's an external/parent contour (parent == -1)
                if area > MIN_AREA and hierarchy[0][i][3] == -1:
                    # Count how many child contours have 'i' as their parent
                    child_count = sum(1 for h in hierarchy[0] if h[3] == i)
                    
                    # If it has more than 3 holes, it's likely the lattice block
                    if child_count > 3:
                        draw_detection(frame_hierarchy, c, f"Holes: {child_count}", (255, 0, 0))


        # =====================================================================
        # TECHNIQUE 3: Morphological Thickness Separation
        # Target: Subtract thick objects (grippers) to find thin objects (lattice)
        # =====================================================================
        # 1. Erode heavily to remove thin lattice struts, keeping only thick grippers
        kernel = np.ones((9, 9), np.uint8)
        thick_objects_mask = cv2.erode(mask, kernel, iterations=1)
        thick_objects_mask = cv2.dilate(thick_objects_mask, kernel, iterations=1) # Restore size slightly
        
        # 2. Subtract thick objects from the original mask
        thin_objects_mask = cv2.subtract(mask, thick_objects_mask)
        
        # 3. Find contours on the remaining (thin) objects
        contours_morph, _ = cv2.findContours(thin_objects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours_morph:
            area = cv2.contourArea(c)
            # Area threshold might need to be lower here since we eroded some of the struts
            if area > MIN_AREA * 0.5: 
                draw_detection(frame_morphology, c, "Morphology", (0, 0, 255))


        # --- DISPLAY WINDOWS ---
        cv2.imshow('Technique 1: Solidity', frame_solidity)
        cv2.imshow('Technique 2: Hierarchy (Holes)', frame_hierarchy)
        # We also show the morphology mask itself so you can understand what the math is doing
        cv2.imshow('Technique 3: Morphology (Mask)', thin_objects_mask) 
        cv2.imshow('Technique 3: Morphology (Result)', frame_morphology)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def draw_detection(frame, contour, text, color):
    """Helper function to draw bounding box, centroid, and text."""
    # 1. Bounding Box
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # 2. Centroid (Pixel Location) using Image Moments
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Draw center point
        cv2.circle(frame, (cX, cY), 5, color, -1)
        
        # Write Coordinates and Label
        label = f"{text} | X:{cX} Y:{cY}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
    main()