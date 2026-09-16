# RingsVision Development Log

## 16 September 2026 — Project Setup and Image-Loading Pipeline

### Objective

Establish the project structure and create a reliable image-loading pipeline before introducing pose estimation.

### Work Completed

* Created and activated a Python virtual environment
* Installed OpenCV and NumPy
* Loaded a static Iron Cross image using `cv2.imread()`
* Added handling for an invalid or unreadable image
* Extracted the image dimensions from its NumPy array
* Converted the image from BGR to RGB
* Displayed the original image using OpenCV
* Reorganised the project directories
* Replaced the machine-specific image path with a portable `pathlib` path
* Connected the local project to the existing GitHub repository
* Integrated the MediaPipe Tasks Pose Landmarker with the Full model
* Refactored image loading and pose detection into separate functions
* Converted normalised landmark coordinates into image-pixel coordinates

### Technical Decisions

* BGR and RGB versions are stored separately because OpenCV uses BGR while MediaPipe expects RGB
* `pathlib` constructs the image path relative to the project folder instead of relying on a machine-specific absolute path
* Sample media is excluded from Git because its redistribution rights have not been confirmed
* Image loading is checked first because `cv2.imread()` returns `None` when it fails, which would cause later image operations to produce errors
* Used IMAGE running mode because the current input is a static image
* Kept the downloaded model outside Git while storing its path relative to the project
* Used MediaPipe’s official pose connections rather than defining the skeleton manually
* Drew on a copy to preserve the original image
* Drew lines before points so landmarks remain visible

### Testing

* Valid filename: the image loaded, its dimensions were printed and the display window opened
* Invalid filename: a descriptive error appeared and the program ended without crashing
* Detected one pose and retrieved 33 landmarks from the test image
* Visually confirmed that the detected landmarks and connections follow the gymnast’s pose

### Problems and Resolutions

* PowerShell initially blocked the virtual-environment activation script, so the execution policy for the current user was changed to `RemoteSigned`
* Reorganising the folders broke the original absolute image path, so it was replaced with a path constructed relative to the project root
* The code expected an `iron_cross` folder while the folder was named `iron cross`, so the folder was renamed
* The local folder was not connected to the existing GitHub repository, so Git was initialised locally and linked to the remote repository

### What I Learned

* NumPy represents an image using the order `(height, width, channels)`
* OpenCV loads colour images in BGR channel order
* MediaPipe expects images in RGB channel order
* `cv2.imread()` returns `None` when an image cannot be read
* Project-relative paths allow the repository to work outside its original location
* A GitHub repository and a local Git repository must be connected before changes can be exchanged

### Next Step

Extract the shoulder, elbow and wrist landmarks and calculate the elbow angles for both arms.