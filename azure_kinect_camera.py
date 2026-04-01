import cv2
import numpy as np
from pyk4a import (
    FPS as K4AFPS,
    CalibrationType,
    ColorResolution,
    Config as K4AConfig,
    DepthMode,
    PyK4A,
    connected_device_count,
)


AZURE_KINECT_RESOLUTION_MAP = {
    (1280, 720): ColorResolution.RES_720P,
    (1920, 1080): ColorResolution.RES_1080P,
    (2560, 1440): ColorResolution.RES_1440P,
    (2048, 1536): ColorResolution.RES_1536P,
    (3840, 2160): ColorResolution.RES_2160P,
    (4096, 3072): ColorResolution.RES_3072P,
}

AZURE_KINECT_FPS_MAP = {
    5: K4AFPS.FPS_5,
    15: K4AFPS.FPS_15,
    30: K4AFPS.FPS_30,
}


def supported_resolutions_text():
    return ", ".join(f"{w}x{h}" for w, h in AZURE_KINECT_RESOLUTION_MAP)


def list_azure_kinect_devices():
    devices = []
    for device_id in range(connected_device_count()):
        camera = PyK4A(device_id=device_id)
        camera.open()
        try:
            devices.append(
                {
                    "device_id": device_id,
                    "serial": camera.serial,
                    "name": f"Azure Kinect DK #{device_id}",
                }
            )
        finally:
            camera.close()
    return devices


def select_azure_kinect_device(requested_serial):
    devices = list_azure_kinect_devices()
    if not devices:
        raise RuntimeError("No Azure Kinect devices found.")

    if requested_serial in (None, "", "none", "None"):
        device = devices[0]
        print(f"Using first Azure Kinect: {device['name']} [{device['serial']}]")
        return device

    for device in devices:
        if device["serial"] == requested_serial:
            print(f"Using requested Azure Kinect: {device['name']} [{device['serial']}]")
            return device

    available = ", ".join(d["serial"] for d in devices)
    raise RuntimeError(
        f"Requested Azure Kinect serial '{requested_serial}' was not found. "
        f"Available serials: {available}"
    )


def start_azure_kinect(device_id, width, height, fps):
    resolution = AZURE_KINECT_RESOLUTION_MAP.get((width, height))
    fps_enum = AZURE_KINECT_FPS_MAP.get(fps)
    if resolution is None:
        raise ValueError(
            f"Azure Kinect color resolution {width}x{height} is not supported. "
            f"Supported values: {supported_resolutions_text()}"
        )
    if fps_enum is None:
        raise ValueError("Azure Kinect supports only 5, 15, or 30 FPS.")

    camera = PyK4A(
        config=K4AConfig(
            color_resolution=resolution,
            camera_fps=fps_enum,
            depth_mode=DepthMode.OFF,
            synchronized_images_only=False,
        ),
        device_id=device_id,
    )
    camera.start()
    return camera


def get_bgr_frame(camera):
    capture = camera.get_capture()
    color = capture.color
    if color is None:
        return None
    if color.ndim == 3 and color.shape[2] == 4:
        return cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
    return color.copy()


def get_camera_intrinsics(camera):
    calibration = camera.calibration
    camera_matrix = calibration.get_camera_matrix(CalibrationType.COLOR).astype(np.float32)
    dist_coeffs = calibration.get_distortion_coefficients(CalibrationType.COLOR).astype(np.float32)
    return camera_matrix, dist_coeffs
