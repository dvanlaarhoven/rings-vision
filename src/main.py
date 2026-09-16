from pathlib import Path

import cv2

# Load and inspect a static Iron Cross image before pose estimation

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "samples"
    / "images"
    / "iron_cross"
    / "IronCross_000.jpg"
)

# Load the iron cross image
iron_cross = cv2.imread(str(IMAGE_PATH))

if iron_cross is None:
    print(f"Error: Unable to read image at {IMAGE_PATH}")
else:
    # MediaPipe expects RGB, while OpenCV loads images in BGR order
    iron_cross_rgb = cv2.cvtColor(iron_cross, cv2.COLOR_BGR2RGB)

    height, width, channels = iron_cross.shape

    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")

    # Display the iron cross image in a window
    cv2.imshow("Iron Cross Exemplar", iron_cross)

    print(f"Successfully loaded an image with dimensions {width} x {height} pixels")

    # Wait for a key press before closing the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()