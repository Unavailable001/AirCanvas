# AirCanvas

# Virtual canvas with Hand Gestures 🎨

A real-time virtual Writing application that lets you draw or write on screen using hand gestures captured through your webcam. Built with OpenCV and MediaPipe, this application enables a natural and interactive drawing experience without physical input devices.

This is an AI-powered Virtual Drawing Canvas project that lets users draw or write on a digital canvas using hand gestures captured through a webcam. Here's a comprehensive breakdown:

Key Features:
1. Drawing Tools:
- Multiple color options (Blue, Green, Red, Yellow)
- Adjustable brush size
- Eraser functionality
- Clear canvas option
- Save Option
- Upload Background
- Pen Mode

2. Hand Gesture Controls:
- Uses MediaPipe for hand landmark detection
- Draws when index finger and thumb are in specific positions
- Gesture-based color selection and tool switching
- Brush size control through virtual buttons

3. Interface Elements:
- Split-screen display showing:
  * Camera feed with hand tracking
  * Drawing canvas
- Color selection buttons
- Tool control buttons
- Brush size controls with plus/minus buttons

4. Additional Features:
- Save functionality to store drawings
- Upload feature to add custom backgrounds
- Pen mode toggle for different drawing styles
- Real-time visualization of hand tracking
- Multiple save slots for different drawings

Technical Implementation:
1. Libraries Used:
- OpenCV (cv2) for image processing and display
- MediaPipe for hand tracking
- NumPy for array operations
- Tkinter for file dialogs
- Collections.deque for point tracking

2. Drawing Mechanics:
- Tracks hand landmarks using MediaPipe
- Uses deque data structures to store drawing points
- Implements different color channels
- Supports variable brush sizes
- Provides eraser functionality with adjustable size

3. User Interface:
- Interactive buttons for tool selection
- Visual feedback for selected tools
- Real-time display of brush size
- Clear option to reset the canvas
- Save and upload functionality for file management

This project creates an interactive and intuitive drawing experience by combining computer vision and hand gesture recognition, allowing users to create digital artwork without traditional input devices like a mouse or stylus.
