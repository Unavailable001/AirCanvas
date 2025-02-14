import cv2
import numpy as np
import mediapipe as mp 
from collections import deque
import os  # for file handling
import tkinter as tk
from tkinter import filedialog
import time


# Initialize deques for different colors
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]
epoints = [deque(maxlen=512)]
pen_points = [deque(maxlen=512)]  # New deque for storing pen mode points

# Indices for tracking which points are being drawn
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
eraser_index = 0
pen_index = 0  #  it's here for reference


# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

# Color definitions
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
colorIndex = 0

# Add brush size variable
brush_size = 2  # Default brush size

# Button colors
clear_color = (220, 220, 220) # Light gray
blue_button = (255, 0, 0)      # Blue
green_button = (0, 255, 0)     # Green
red_button = (0, 0, 255)       # Red
yellow_button = (0, 255, 255)  # Yellow
exit_color = (0, 0, 128)       # Dark red
eraser_color = (128, 128, 128)  # Gray
save_color = (255, 50, 200)  # purple for save button
upload_color = (200, 255, 200)  # Light gray for the upload button
pen_mode_color = (255, 150, 100)  # sky for pen button

# Initialize pen mode,upload mode and save
pen_mode = True
uploaded= False
saved=0

# Setup paint window
paintWindow = np.zeros((471, 636, 3)) + 255

# Function to draw circular buttons with plus and minus signs
def draw_size_controls(image):
    # Draw plus button
    cv2.circle(image, (30, 150), 15, (200, 200, 200), -1)  # Filled circle
    cv2.circle(image, (30, 150), 15, (0, 0, 0), 1)  # Border
    # Draw plus sign
    cv2.line(image, (23, 150), (37, 150), (0, 0, 0), 2)  # Horizontal
    cv2.line(image, (30, 143), (30, 157), (0, 0, 0), 2)  # Vertical
    
    # Draw minus button
    cv2.circle(image, (30, 200), 15, (200, 200, 200), -1)  # Filled circle
    cv2.circle(image, (30, 200), 15, (0, 0, 0), 1)  # Border
    # Draw minus sign
    cv2.line(image, (23, 200), (37, 200), (0, 0, 0), 2)
    
    # Draw current brush size indicator
    cv2.putText(image, f"Size: {brush_size}", (10, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Draw a preview circle showing current brush size
    cv2.circle(image, (30, 280), brush_size, (0, 0, 0), -1)

# Draw colored buttons on paint window
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), clear_color, -1)    # Clear button filled
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), blue_button, -1)   # Blue button filled
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), green_button, -1)  # Green button filled
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), red_button, -1)    # Red button filled
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), yellow_button, -1) # Yellow button filled
paintWindow = cv2.rectangle(paintWindow, (40, 406), (140, 470), exit_color, -1)  # Exit button filled
paintWindow = cv2.rectangle(paintWindow, (160, 406), (255, 470), eraser_color, -1) # Eraser button filled
paintWindow = cv2.rectangle(paintWindow, (280, 406), (370, 470), save_color, -1)  # Save button filled
paintWindow = cv2.rectangle(paintWindow, (400, 406), (500, 470), upload_color, -1)  # Upload button filled
paintWindow = cv2.rectangle(paintWindow, (520, 406), (620, 470), pen_mode_color, -1)


# Draw button borders
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (40, 406), (140, 470), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 406), (255, 470), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (280, 406), (370, 470), (0, 0, 0), 2)    # Button border
paintWindow = cv2.rectangle(paintWindow, (400, 406), (500, 470), (0, 0, 0), 2)      # Button border
paintWindow = cv2.rectangle(paintWindow, (520, 406), (620, 470), (0, 0, 0), 2)


# Add text with contrasting colors
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "ERASER", (180, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "EXIT(q)", (60, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "SAVE", (300, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "UPLOAD", (420, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "PEN MODE", (530, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw colored buttons on the frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), clear_color, -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), blue_button, -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), green_button, -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), red_button, -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), yellow_button, -1)
    frame = cv2.rectangle(frame, (40, 406), (140, 470), exit_color, -1)
    frame = cv2.rectangle(frame, (160, 406), (255, 470), eraser_color, -1)
    frame = cv2.rectangle(frame, (280, 406), (370, 470), save_color, -1)  # Save button filled
    frame = cv2.rectangle(frame, (400, 406), (500, 470), upload_color, -1)  # Upload button filled
    frame = cv2.rectangle(frame, (520, 406), (620, 470), pen_mode_color, -1)


    # Draw button borders
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (40, 406), (140, 470), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 406), (255, 470), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (280, 406), (370, 470), (0, 0, 0), 2)    # Button border
    frame = cv2.rectangle(frame, (400, 406), (500, 470), (0, 0, 0), 2)      # Button border
    frame = cv2.rectangle(frame, (520, 406), (620, 470), (0, 0, 0), 2)
 

    # Add text with contrasting colors
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "ERASER", (180, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "EXIT", (60, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "SAVE", (300, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "UPLOAD", (420, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "PEN MODE", (530, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    
    # Draw the size control buttons
    draw_size_controls(frame)
    draw_size_controls(paintWindow)

    # Get hand landmark prediction
    result = hands.process(framergb)

     
    # Background image variable
    # background = np.ones((471, 636, 3), np.uint8) * 255  # Default white background
    
    # Initialize tkinter root for file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    
    
    # Post-process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        
        # Get coordinates of index finger and thumb  
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        
        # Print the coordinates to the terminal
        # print(f"Index Finger: {fore_finger}, Thumb: {thumb}")
        # For printing Coordinates
        print("Coordinates:", center)
          
        fingers_distance = abs(thumb[1] - center[1])
        fingers_are_close = fingers_distance < 35
        
        # Check if the thumb is close to the index finger
        if (pen_mode and fingers_are_close) or (not pen_mode and not fingers_are_close):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1
            epoints.append(deque(maxlen=512))
            eraser_index += 1
        
        # Update the brush size control detection
        if 15 <= center[0] <= 45:  # X-coordinate range for size buttons
              if 135 <= center[1] <= 165:  # Y-coordinate range for plus button
                time.sleep(0.06)
                brush_size = min(20, brush_size + 1)  # Increase size with upper limit
                print(f"Brush size increased to: {brush_size}")  # Debug print
              elif 185 <= center[1] <= 215:  # Y-coordinate range for minus button
                time.sleep(0.06)
                brush_size = max(1, brush_size - 1)  # Decrease size with lower limit
                print(f"Brush size decreased to: {brush_size}")  # Debug print

        
        elif center[1] <= 65:
            # Clear button
            if 40 <= center[0] <= 140:
                print("Clearing Window")
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                epoints = [deque(maxlen=512)]
               
                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0
                eraser_index = 0
                #clear the paint window
                if uploaded : paintWindow = background.copy()
                else : paintWindow[67:405,:,:]=255 #clear the particuar painting zone
                   #paintWindow[:,:,:]=255 #clear the whole paint window
            # Color selection buttons
            elif 160 <= center[0] <= 255:
                colorIndex = 0  # Blue
                print("Blue Selected")
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # Green
                print("Green Selected")
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # Red
                print("Red selected")
            elif 505 <= center[0] <= 600:
                colorIndex = 3  # Yellow
                print("Yellow selected")

        elif 406 <= center[1] <= 470:
            # Eraser and Exit button logic
            if 160 <= center[0] <= 255:
                colorIndex = 4  # Eraser
                cv2.putText(frame,"SELECTED", (160, 380), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                print("Eraser Selected")
            elif 40 <= center[0] <= 140:
                break  # Exit application
                print("Exit")
             # Save button logic  
            elif 280 <= center[0] <= 370:
             saved+=1 
             
             filename = "DrawingCanvas.png"
             filename2= "CameraWindow.png"
             
             #for Multiple saving
             if saved>1 :
              filename = f"DrawingCanvas{saved}.png"
              filename2= f"CameraWindow{saved}.png"
             
             time.sleep(0.5)
             
             cv2.imwrite(filename, paintWindow)  # Save the Drawing
             
             cv2.imwrite(filename2, frame)  # Save the Camera window
             
             cv2.putText(frame,f"SAVED {saved}", (280, 380), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
             print("Saving")
             
            elif 400 <= center[0] <= 500:
               # Open a file dialog to select an image
                print("Uploading")
                uploaded=True
                filepath = filedialog.askopenfilename(title="Select Background Image", 
                                               filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
                
                if filepath:  # Check if a file was selected
                 background = cv2.imread(filepath)  # Read the selected image
                 background = cv2.resize(background, (636, 471))  # Resize to fit the window
                 
                 paintWindow = background.copy()  # Set the paintWindow to the new background
                else:
                 print("Error: Unable to load the image. Please check the file format.")
     
            elif 520 <= center[0] <= 620:  # Pen Mode button
                    print("Pen Mode")
                    time.sleep(0.4)
                    pen_mode = not pen_mode
                    # Visual feedback for pen mode
                    pen_status = "PEN MODE: ON" if pen_mode else "PEN MODE: OFF"
                    cv2.putText(frame, pen_status, (520, 380), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        else:
           if colorIndex == 4:
                # Eraser functionality
                eraser_radius = 20
                
                # Function to check if a point should be erased
                def should_erase(point):
                    return np.linalg.norm(np.array(point) - np.array(center)) < eraser_radius if point is not None else False
                
                # Process each color's points
                for i in range(len(bpoints)):
                    bpoints[i] = deque([p for p in bpoints[i] if not should_erase(p)], maxlen=512)
                for i in range(len(gpoints)):
                    gpoints[i] = deque([p for p in gpoints[i] if not should_erase(p)], maxlen=512)
                for i in range(len(rpoints)):
                    rpoints[i] = deque([p for p in rpoints[i] if not should_erase(p)], maxlen=512)
                for i in range(len(ypoints)):
                    ypoints[i] = deque([p for p in ypoints[i] if not should_erase(p)], maxlen=512)
                    
                if uploaded :  paintWindow = background.copy()
                # Draw eraser effect on both windows
                cv2.circle(frame, center, eraser_radius, (255, 255, 255), -1)
                cv2.circle(paintWindow, center, eraser_radius, (255, 255, 255), -1)
           else:
                # Only append points if not in eraser mode
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)

    else:
        # Only create new deques if the index is not in eraser mode
        if colorIndex != 4:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1
            epoints.append(deque(maxlen=512))
            eraser_index += 1
    
    # Draw lines of all the colors
    points = [bpoints, gpoints, rpoints, ypoints, epoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            points_list = list(points[i][j])
            for k in range(1, len(points_list)):
                if points_list[k - 1] is not None and points_list[k] is not None:
                    cv2.line(frame, points_list[k - 1], points_list[k], colors[i], brush_size)
                    cv2.line(paintWindow, points_list[k - 1], points_list[k], colors[i], brush_size)
                if i == 4:  # Eraser
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[4], 50)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[4], 50)
                else:
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)


    cv2.imshow("Frame", frame)
    #paintWindow = cv2.addWeighted(background, 1, paintWindow, 0.5, 0)  # Combine background with the drawing
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break


# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()