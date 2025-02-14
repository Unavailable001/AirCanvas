# AirCanvas

# Virtual canvas with Hand Gestures 🎨

A real-time virtual Writing application that lets you draw or write on screen using hand gestures captured through your webcam. Built with OpenCV and MediaPipe, this application enables a natural and interactive drawing experience without physical input devices.

## Features ✨

- **Color Selection**: Choose from multiple colors including:
  - Blue
  - Green
  - Red
  - Yellow

- **Drawing Tools**:
  - Adjustable brush size (1-20 pixels)
  - Eraser tool
  - Clear canvas option
  - Pen mode toggle for precise control

- **Additional Functionality**:
  - Save your artwork
  - Upload background images
  - Real-time webcam feed with hand tracking
  - Intuitive gesture-based controls

## Requirements 📋

```
python >= 3.7
opencv-python
numpy
mediapipe
tkinter (usually comes with Python)
```

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/virtual-paint.git
cd virtual-paint
```

2. Install required packages:
```bash
pip install opencv-python numpy mediapipe
```

## Usage 🖌️

1. Run the application:
```bash
python virtual_paint.py
```

2. The application will open two windows:
   - Camera feed window with control interface
   - Paint canvas window

3. **Controls**:
   - Use your index finger to interact with buttons
   - Drawing modes:
     - Pen Mode ON: Draw when index finger and thumb are close
     - Pen Mode OFF: Draw when fingers are apart
   - Adjust brush size using + and - buttons on the left side
   - Select colors from the top toolbar
   - Access utilities from the bottom toolbar

4. **Button Layout**:
   - Top Row: Clear, Blue, Green, Red, Yellow
   - Bottom Row: Exit, Eraser, Save, Upload, Pen Mode

## Hand Gestures Guide 👆

- **Drawing**: 
  - In Pen Mode ON: Bring thumb and index finger close together
  - In Pen Mode OFF: Keep fingers apart
- **Color Selection**: Point index finger at color buttons
- **Brush Size**: Use index finger to click + or - buttons
- **Clear Canvas**: Point to Clear button
- **Save**: Point to Save button
- **Exit**: Point to Exit button or press 'q'

## Save Features 💾

- Saved drawings are stored as PNG files:
  - `DrawingCanvas.png`: Your artwork
  - `CameraWindow.png`: Screenshot of the camera feed
- Multiple saves create numbered files (e.g., `DrawingCanvas1.png`, `DrawingCanvas2.png`)

## Background Upload 🖼️

1. Click the Upload button
2. Select an image file (supported formats: JPG, JPEG, PNG)
3. The selected image becomes your canvas background

## Tips for Best Results 💡

1. Ensure good lighting conditions
2. Keep your hand clearly visible to the camera
3. Maintain a steady distance from the camera
4. Use slow, deliberate movements for precise drawing
5. Experiment with Pen Mode ON/OFF to find your preferred drawing style


## Acknowledgments 🙏

- OpenCV for computer vision capabilities
- MediaPipe for hand tracking technology
- Original inspiration from virtual drawing and writing systems
