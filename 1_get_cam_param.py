import cv2
import numpy as np
import glob
import os
import argparse
import yaml

def parse_args():
    p = argparse.ArgumentParser(
        description="Chessboard Camera Calibration Tool")
    p.add_argument('-i', '--input',    type=str, default='img/',
                   help='Input folder with calibration images')
    p.add_argument('-w', '--width',    type=int, default=4,
                   help='Number of inner corners per chessboard row')
    p.add_argument('-H', '--height',   type=int, default=6,
                   help='Number of inner corners per chessboard column')
    p.add_argument('-s', '--square',   type=float, default=40.0,
                   help='Square size in real-world units (mm)')
    p.add_argument('--remove-bad',     action='store_true',
                   help='Delete images where pattern not found')
    p.add_argument('-o', '--output',   type=str, default='camIntrinsic.yaml',
                   help='Output file for camera matrix & distortion coeffs')
    return p.parse_args()

def collect_image_points(images, board_size, square_size, remove_bad):
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                 30, 0.001)
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0],
                           0:board_size[1]].T.reshape(-1, 2) * square_size

    objpoints, imgpoints = [], []
    valid_images = []
    for fname in sorted(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size)
        if not found:
            print(f"[WARN] Pattern not found in {os.path.basename(fname)}")
            if remove_bad:
                os.remove(fname)
                print(f"       Deleted {os.path.basename(fname)}")
            continue

        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term_crit)
        objpoints.append(objp)
        imgpoints.append(corners)
        valid_images.append(fname)

        # 시각 확인
        disp = cv2.drawChessboardCorners(img, board_size, corners, found)
        cv2.imshow('Corners', disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow('Corners')
        if key == ord('q'):
            break

    return objpoints, imgpoints, gray.shape[::-1], valid_images

def calibrate_and_save(objpoints, imgpoints, image_size, output_file):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    print("\n[RESULT] RMS re-projection error:", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist.ravel())

    data = {
        'camera_matrix': mtx.tolist(),
        'dist_coeff': dist.ravel().tolist(),
        'image_size': list(image_size)
    }
    with open(output_file, 'w') as f:
        yaml.dump(data, f)
    print(f"\nSaved calibration to '{output_file}'")

def show_undistort(images, mtx, dist):
    for fname in images:
        img = cv2.imread(fname)
        und = cv2.undistort(img, mtx, dist, None, mtx)
        combo = np.hstack((img, und))
        cv2.imshow('Original | Undistorted', combo)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow('Original | Undistorted')
        if key == ord('q'):
            break

def main():
    args = parse_args()
    imgs = glob.glob(os.path.join(args.input, '*.jpg'))
    if not imgs:
        print(f"[ERROR] No images found in '{args.input}'")
        return

    board = (args.width, args.height)
    objp, imgp, imsize, valid = collect_image_points(
        imgs, board, args.square, args.remove_bad)
    if not objp:
        print("[ERROR] No valid calibration pairs collected.")
        return

    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objp, imgp, imsize, None, None)
    calibrate_and_save(objp, imgp, imsize, args.output)
    show_undistort(valid, mtx, dist)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# # 기본: img/ 폴더의 4×6, 40mm 체스보드 이미지로 캘리브레이션 수행
# python calib.py

# # 리무브 옵션 켜기
# python calib.py --remove-bad

# # 이미지 폴더, 보드 사이즈, 스퀘어 사이즈, 출력 파일 지정
# python calib.py -i captures/ -w 9 -h 6 -s 25.0 -o mycalib.yaml