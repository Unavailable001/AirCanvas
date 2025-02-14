import cv2
import numpy as np
import mediapipe as mp 
from collections import deque
import os  # for file handling
import tkinter as tk
from tkinter import filedialog
import time
import streamlit as st


# Streamlit page configuration
st.set_page_config(page_title="OpenCV with Streamlit", layout="wide")


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
cv2.putText(paintWindow, "EXIT", (60, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
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


def process_frame():
    
    global saved
    global blue_index, green_index, red_index, yellow_index
    global bpoints, gpoints, rpoints, ypoints, colorIndex 
    global paintWindow ,epoints ,eraser_index,pen_mode,brush_size,uploaded
    
    
    ret, frame = cap.read()
    if not ret:
        return None, None, False

    x, y, c = frame.shape
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
    frame = cv2.rectangle(frame, (280, 406), (370, 470), (0, 0, 0), 2)      # Button border
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
                if uploaded : 
                    paintWindow = background.copy()
                else : #paintWindow[67:405,:,:]=255 #clear the particuar painting zone
                   paintWindow[:,:,:]=255 #clear the whole paint window
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
                # Clear the Streamlit windows
                print("Exit")
                stframe.empty()
                stpaint.empty()
                cap.release()
                return frame, paintWindow, True  # Exit condition is met
                
             # Save button logic  
            elif 280 <= center[0] <= 370:
             saved+=1 
             
             filename = "DrawingCanvas.png"
             filename2= "CameraWindow.png"
             
             #for Multiple saving
             if saved>1 :
              filename = f"DrawingCanvas{saved}.png"
              filename2= f"CameraWindow{saved}.png"
             
             time.sleep(0.4)
             
             cv2.imwrite(filename, paintWindow)  # Save the Drawing
             
             cv2.imwrite(filename2, frame)  # Save the Camera window
             
             cv2.putText(frame,f"SAVED {saved}", (280, 380), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
             print("Saving")
             
            elif 400 <= center[0] <= 500:
              '''' # Open a file dialog to select an image
                print("Uploading")
                uploaded=True
                filepath = filedialog.askopenfilename(title="Select Background Image", 
                                               filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
                
                if filepath:  # Check if a file was selected
                 background = cv2.imread(filepath)  # Read the selected image
                 background = cv2.resize(background, (636, 471))  # Resize to fit the window
                 
                 paintWindow = background.copy()  # Set the paintWindow to the new background
                else:
                 print("Error: Unable to load the image. Please check the file format.")''' 
              with st.sidebar.expander("Upload Background Image"):
                  uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
                  if uploaded_file is not None:
                   bytes_data = uploaded_file.getvalue()
                   nparr = np.frombuffer(bytes_data, np.uint8)
                   background = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                   background = cv2.resize(background, (636, 471))
                   paintWindow = background.copy()
                   uploaded = True
                  else:
                      uploaded = False
     
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
    
    
    return frame, paintWindow, False
    

# Navigation bar
st.markdown("""
<style>
    body, nav, footer, .stButton, .stSelectbox, .stTextInput, .stTextArea, .stRadio, .stSidebar {
        font-family: 'Arial', sans-serif;
    }
    nav a {
        text-decoration: none;
        color: white !important;  /* Change text color to white */
        font-size: 20px;
        margin-right: 20px;
        transition: transform 0.3s, color 0.3s; /* Smooth transition for transform and color */
    }
    nav a:hover {
        transform: scale(1.1); /* Scale up the text */
        color: cyan !important;  /* Change text color on hover */
    }
</style>
""", unsafe_allow_html=True)

# Navigation bar
st.markdown("""
<nav style="background-color:  grey; padding: 10px;">
    <a style="margin-right: 20px; font-size: 20px;" href="#welcome">Home</a>
    <a style="margin-right: 20px; font-size: 20px;" href="#try-out-canvas">Try Canvas</a>
    <a style="margin-right: 20px; font-size: 20px;" href="#used-libraries-in-python">Tools used</a>
    <a style="margin-right: 20px; font-size: 20px;" href="#applications-of-the-project">Applications</a>
    <a style="margin-right: 20px; font-size: 20px;" href="#project-description">Project Description</a>
    <a style="font-size: 20px;" href="#contact-us">Contact</a>
</nav>
""", unsafe_allow_html=True)

# Welcome section
st.text(" ")
st.title("Welcome to The Project")
st.image("https://th.bing.com/th/id/R.98ab810f871847d63e8ea5a9bb0f5c58?rik=PXtGZdxCHRb55w&riu=http%3a%2f%2fwww.sightwords.com%2fimages%2fmath%2fcounting%2fair_writing.jpg&ehk=bTlt4F6iJtm%2bIAyhpsuYPHi8H3Pzo%2bbB9meUB5K1kVs%3d&risl=&pid=ImgRaw&r=0")
st.title("Air Canvas")
st.write("""
    Air Canvas is an innovative application that allows users to draw or write in the air using hand gestures, leveraging the power of computer vision and real-time tracking. Using a webcam or device camera, the application tracks the movement of the user's hand, transforming it into a virtual pen for creative expression in mid-air.
    Whether you're sketching, doodling, or signing, Air Canvas offers a fun and intuitive way to create visual art without the need for traditional input devices. Built with cutting-edge technologies like OpenCV and MediaPipe, Air Canvas provides an engaging and hands-free drawing experience.
    The app recognizes hand gestures to select different colors, erase drawings, and even clear the canvas with a simple swipe. It's perfect for users who want to explore drawing in a dynamic, interactive way, all while promoting creativity and enhancing motor skills through gesture-based interaction.
    Ideal for both casual users and creative professionals, Air Canvas opens up new possibilities for artistic expression in digital spaces.
""")

st.title("Used Libraries in Python")

# descriptions of the libraries
library_descriptions = {
    "OpenCV": (
        "OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. "
        "It contains more than 2500 optimized algorithms, which can be used for a wide range of tasks including: detecting and recognizing faces, "
        "identifying objects, classifying human actions in videos, tracking camera movements, extracting 3D models of objects, stitching images together "
        "to produce a high-resolution image of an entire scene, finding similar images from an image database, removing red-eye, following eye movements, "
        "recognizing scenery and establishing markers to overlay it with augmented reality, etc."
    ),
    "Numpy": (
        "NumPy is a core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working "
        "with these arrays. NumPy offers comprehensive mathematical functions, random number generators, linear algebra routines, Fourier transforms, "
        "and more. It is the foundation on which nearly all other scientific and numerical libraries in Python, such as SciPy, pandas, and scikit-learn, "
        "are built."
    ),
    "MediaPipe": (
        "MediaPipe is a cross-platform framework for building multimodal (e.g., video, audio, etc.) machine learning pipelines. It is designed to help "
        "developers build perception pipelines, such as those for real-time face detection, hand tracking, and pose estimation, with ease and flexibility. "
        "MediaPipe offers ready-to-use solutions and tools that can be integrated into applications to process and analyze multimedia data efficiently."
    ),
    "Dequeue": (
        "A deque (double-ended queue) is an ordered collection of items similar to the standard Python list. The main feature of a deque is that it allows "
        "you to add and remove elements from both ends—either from the front or the back—with O(1) time complexity for append and pop operations. Deques "
        "are a part of the collections module and are useful for implementing stacks, queues, and other data structures where you need efficient insertions "
        "and deletions from both ends."
    ),
    "streamlit": (
        "Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. "
        "With Streamlit, you can build interactive dashboards and apps with a few lines of code, allowing you to visualize and explore data, display model "
        "outputs, and create user interfaces for your data science projects. It supports various widgets for user input, real-time updates, and seamless "
        "integration with other Python libraries."
    ),
    "os": (
        "The os module in Python provides a way of using operating system dependent functionality like reading or writing to the file system. "
        "It allows you to interface with the underlying operating system that Python is running on, such as creating, removing, and changing directories, "
        "executing shell commands, and manipulating environment variables."
    ),
    "tkinter": (
        "Tkinter is the standard GUI (Graphical User Interface) library for Python. It provides a powerful object-oriented interface to the Tk GUI toolkit. "
        "With Tkinter, you can create windows, dialogs, buttons, menus, and other GUI components for your applications. It is a versatile library that "
        "supports various widgets for building desktop applications."
    ),
    "filedialog": (
        "The filedialog module is part of the tkinter library and provides classes and factory functions for creating file/directory selection windows. "
        "It allows users to select files or directories through a standard dialog interface, making it easy to implement file opening, saving, and "
        "directory selection features in your applications."
    ),
    "time": (
        "The time module provides various time-related functions. You can use it to handle time-related tasks such as getting the current time, "
        "pausing the execution of a program (sleeping), measuring the time taken by a piece of code to execute, and formatting time for display. "
        "It is useful for performance testing, scheduling, and working with timestamps."
    )
}

# Create a select box with the library options
selected_library = st.selectbox("Select a library to learn more about it:", ["OpenCV", "Numpy", "MediaPipe", "Dequeue", "streamlit", "os", "tkinter", "filedialog", "time"])
# Display the description of the selected library
if selected_library:
    st.write(library_descriptions[selected_library])


st.title("Applications of the Project")

st.write("""
The project focuses on solving some major societal problems:

1. **People hearing impairment**: Although we take hearing and listening for granted, people with hearing impairment communicate using sign languages. Most of the world can't understand their feelings and emotions without a translator in between.
2. **Overuse of Smartphones**: Smartphones cause accidents, depression, distractions, and other illnesses that humans are still discovering. Although their portability and ease of use are profoundly admired, the negatives include life-threatening events.
3. **Paper wastage**: We waste a lot of paper in scribbling, writing, drawing, etc. Some basic facts include - 5 liters of water on average are required to make one A4 size paper, 93% of writing is from trees, 50% of business waste is paper, 25% landfill is paper, and the list goes on. Paper wastage is harming the environment by using water and trees and creating tons of garbage.

4. **Educational Tools**:
   - **Interactive Learning**: Air Writing can be used in classrooms to make learning more interactive. Teachers can write in the air to explain concepts, and students can participate actively without needing physical materials.
   - **Remote Learning**: During online classes, teachers can use Air Writing to visually explain difficult concepts, making virtual education more effective.
5. **Healthcare**:
   - **Patient Communication**: In hospitals, especially in intensive care units where patients might not be able to speak, Air Writing can be used to communicate needs and feelings to healthcare providers.
   - **Rehabilitation**: For patients undergoing physical rehabilitation, Air Writing can be a therapeutic exercise to improve motor skills and coordination.
6. **Public Safety and Navigation**:
   - **Emergency Communication**: In situations where verbal communication is impossible, such as during a fire or in noisy environments, Air Writing can be used to convey critical information.
   - **Navigation Assistance**: For people with disabilities, Air Writing can help them navigate public spaces by providing written instructions or directions in real-time.
7. **Art and Creativity**:
   - **Digital Art**: Artists can use Air Writing to create digital art pieces, enabling them to draw or write in a 3D space, enhancing creativity and offering new ways to express themselves.
   - **Augmented Reality Exhibits**: Museums and galleries can use Air Writing to create interactive exhibits where visitors can add their own contributions or interact with the displays.
8. **Business and Productivity**:
   - **Virtual Meetings**: During virtual meetings, participants can use Air Writing to jot down notes or highlight important points, making discussions more dynamic and engaging.
   - **Quick Annotations**: Professionals can quickly write reminders or annotations in the air, which can be saved and accessed later, improving productivity and reducing reliance on paper.
9. **Entertainment and Gaming**:
   - **Interactive Games**: Air Writing can be incorporated into augmented reality (AR) games, where players can interact with the game environment by writing or drawing in the air.
   - **Virtual Concerts and Events**: During virtual events, artists can interact with the audience by writing messages in the air, creating a more immersive experience.
10. **Travel and Tourism**:
    - **Tourist Information**: Tour guides can use Air Writing to provide information and directions to tourists in a more engaging way.
    - **Language Translation**: In foreign countries, travelers can write in their native language in the air, and the text can be translated in real-time to the local language, facilitating communication.
11. **Environmental Monitoring**:
    - **Real-Time Data Display**: Environmental scientists and activists can use Air Writing to display real-time data about pollution levels, weather conditions, and other environmental metrics during fieldwork or public demonstrations.

By integrating Air Writing into these various applications, we can address multiple societal issues, making communication more inclusive, reducing environmental impact, and enhancing productivity and creativity across different sectors.

Air Writing can quickly solve these issues. It will act as a communication tool for people with hearing impairment. Their air-written text can be presented using AR or converted to speech. One can quickly write in the air and continue with their work without much distraction. Additionally, writing in the air does not require paper. Everything is stored electronically.
""")


# Try Out Canvas section
st.title("Try Out Canvas")
if __name__ == "__main__":
    run = st.button("Start AirCanvas")

    if run:
        st.header("Camera Window")
        stframe = st.empty()
        st.header("Paint Window")
        stpaint = st.empty()
        stop_flag = False
        stop = st.button("Stop AirCanvas")
        #st.snow()
        while True:
            frame, paintWindow, exit_flag = process_frame()
            if frame is None:
                break
            if exit_flag:
                break

            stframe.image(frame, channels="BGR")
            stpaint.image(paintWindow.astype(np.uint8), channels="BGR")

            if stop:
                stop_flag = True
                break

        if stop_flag:
            cap.release()
            cv2.destroyAllWindows()

# Rating section
st.subheader("Rate Our Project")
rating = st.select_slider(" ", ["Worst","Bad", "Good", "Excellent", "Outstanding"])
st.text_area("Tell Us About Your Experience")
st.balloons()

# Project Description section
st.title("Project Description")
st.write("""
    This project leverages the power of Streamlit to provide an interactive web application for displaying images. It allows users to easily load images from local files, URLs, or dynamically generated image objects.
    With its simple and intuitive interface, users can adjust image display properties like width, providing flexibility for various use cases. Whether you're showcasing images from a local source or linking to external content, this project ensures a seamless visual experience.
    Ideal for developers looking to create image-centric applications with minimal code and maximum functionality.
""")


# Sidebar section
st.sidebar.success("Active")
st.sidebar.title("Private Profile")
st.sidebar.text_input("Mail Address")
st.sidebar.text_input("Password")
st.sidebar.button("Submit")
st.sidebar.radio("Professional Expert", ["Student", "Working", "Others"])
st.sidebar.text_area("Request for Further Features")
st.sidebar.button("Submit", key="2")


# Contact section
st.title("Contact Us")
st.write("If you have any questions or feedback, feel free to reach out at dev.d@email.com.")

# Footer
st.markdown("""
<footer style="background-color: grey; text-align: center;font-size: 30px;padding : 10px">
    <p>© 2024 DEV D's. All Rights Reserved.</p>
</footer>
""", unsafe_allow_html=True)