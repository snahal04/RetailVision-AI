import cv2
import sys
from pathlib import Path

VIDEO_PATH = "videos/Entry_CAM.mp4"
OUTPUT_PATH = "output/frame.jpg"


def extract_first_frame(video_path: str, output_path: str) -> None:
    path = Path(video_path)
    if not path.exists():
        print(f"Error: video file not found at '{video_path}'")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video '{video_path}'")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: could not read first frame from video")
        sys.exit(1)

    success = cv2.imwrite(output_path, frame)
    if not success:
        print(f"Error: could not write frame to '{output_path}'")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Saved first frame to '{output_path}' ({w}x{h})")



def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x={x}, y={y}")



if __name__ == "__main__":
    # extract_first_frame(VIDEO_PATH, OUTPUT_PATH)
    img = cv2.imread("output/frame.jpg")

    cv2.imshow("Frame", img)
    cv2.setMouseCallback("Frame", click)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # x=1286, y=243  - near fire safety
    # x=721, y=816  - near purpple board
    # diagonal line

    