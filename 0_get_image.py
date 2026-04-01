import argparse
import os
import re

import cv2

from azure_kinect_camera import get_bgr_frame, start_azure_kinect, supported_resolutions_text


def parse_resolution(res_str):
    """
    Parse strings like '1280x720' or '1280x720@30' into (width, height, fps).
    """
    match = re.match(r"(\d+)x(\d+)(?:@(\d+))?$", res_str)
    if not match:
        raise argparse.ArgumentTypeError("Resolution must be WIDTHxHEIGHT or WIDTHxHEIGHT@FPS")
    w, h, f = match.groups()
    return int(w), int(h), int(f) if f else None


def get_start_index(output_dir, prefix):
    """
    Scan output_dir for existing files named prefix_<idx>.jpg
    and return next available index.
    """
    files = os.listdir(output_dir)
    idxs = []
    for fname in files:
        m = re.match(rf"{prefix}_(\d+)\.jpg$", fname)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs) + 1 if idxs else 0


def capture_and_save_images(camera, output_dir):
    """
    Main loop: show color stream, overlay instructions,
    capture on 'c', quit on 'q'.
    """
    idx = get_start_index(output_dir, "capture")
    win = "Azure Kinect Capture"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        img = get_bgr_frame(camera)
        if img is None:
            continue

        overlay = img.copy()
        cv2.putText(
            overlay,
            "Press 'c' to capture, 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.imshow(win, img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            orig = os.path.join(output_dir, f"capture_{idx}.jpg")
            cv2.imwrite(orig, img)
            print(f"Saved: {orig}")
            idx += 1
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Azure Kinect Image Capture Tool")
    parser.add_argument(
        "-r",
        "--resolution",
        type=parse_resolution,
        default=(1280, 720, 30),
        help=f"Capture resolution, e.g. '1280x720' or '1280x720@30'. Supported: {supported_resolutions_text()}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="img",
        help="Output directory for saved images",
    )
    args = parser.parse_args()

    w, h, fps = args.resolution
    os.makedirs(args.output, exist_ok=True)

    camera = start_azure_kinect(device_id=0, width=w, height=h, fps=fps or 30)
    try:
        capture_and_save_images(camera, args.output)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
