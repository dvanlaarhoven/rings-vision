from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load and inspect a static Iron Cross image before pose estimation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = (
    PROJECT_ROOT
    / "samples"
    / "images"
    / "iron_cross"
    / "IronCross_000.jpg"
)
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"


def load_image(image_path: Path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image at {image_path}"
        )

    return image


def detect_pose_landmarks(image_rgb, model_path: Path):
    base_options = python.BaseOptions(
        model_asset_path=str(model_path)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    with vision.PoseLandmarker.create_from_options(
        options
    ) as landmarker:
        return landmarker.detect(mp_image)


def main():
    try:
        iron_cross = load_image(IMAGE_PATH)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return

    # MediaPipe expects RGB, while OpenCV loads images in BGR order
    iron_cross_rgb = cv2.cvtColor(iron_cross, cv2.COLOR_BGR2RGB)

    pose_result = detect_pose_landmarks(
        iron_cross_rgb,
        MODEL_PATH,
    )

    pose_count = len(pose_result.pose_landmarks)
    print(f"Poses detected: {pose_count}")

    if pose_count > 0:
        landmark_count = len(pose_result.pose_landmarks[0])
        print(f"Landmarks detected: {landmark_count}")

    height, width, channels = iron_cross.shape

    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")

    # Display the iron cross image in a window
    cv2.imshow("Iron Cross Exemplar", iron_cross)

    print(
        f"Successfully loaded an image with dimensions "
        f"{width} x {height} pixels"
    )

    # Wait for a key press before closing the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()