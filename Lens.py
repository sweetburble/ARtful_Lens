import numpy as np
import cv2 as cv

def select_img_from_video(input_file, board_pattern, select_all=False, wait_msec=10):
    # Open a video
    video = cv.VideoCapture(input_file)
    assert video.isOpened(), 'Cannot read the given input, ' + input_file

    # Select images
    img_select = []
    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow('Camera Calibration', display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == 27:                  # 'ESC' key: Exit (Complete image selection)
                break
            elif key == ord(' '):          # 'Space' key: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow('Camera Calibration', display)
                key = cv.waitKey()
                if key == 27: # ESC
                    break
                elif key == ord('\r'):
                    img_select.append(img) # 'Enter'키는 이미지를 선택한다

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0, 'There is no set of complete chessboard points!'

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be 'np.float32'

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)



if __name__ == '__main__':
    input_file = 'chchch_board.mp4'
    board_pattern = (10, 7)
    board_cellsize = 0.025
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    img_select = select_img_from_video(input_file, board_pattern)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

    # Open a video
    video = cv.VideoCapture(input_file)
    assert video.isOpened(), 'Cannot read the given input, ' + input_file

    # Prepare a 3D box for simple AR
    box_lower = board_cellsize * np.array([[2, 3, 0], [3, 2,  0], [4, 1,  0], [5, 2,  0], [6, 1,  0], [7, 2, 0], [8, 3, 0], [7, 4, 0], [6, 5, 0], [5, 6, 0], [4, 5, 0], [3, 4, 0]]) # 파란색 선
    box_upper = board_cellsize * np.array([[2, 3, -1], [3, 2,  -1], [4, 1,  -1], [5, 2,  -1], [6, 1,  -1], [7, 2, -1], [8, 3, -1], [7, 4, -1], [6, 5, -1], [5, 6, -1], [4, 5, -1], [3, 4, -1]]) # 빨간색 박스

    # Prepare 3D points on a chessboard
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    # Run pose estimation
    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
            break

        # Estimate the camera pose
        complete, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
        if complete:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

            # Draw the box on the image
            line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
            cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)
            # for b, t in zip(line_lower, line_upper):
            #     cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        # Show the image and process the key event
        cv.imshow('Pose Estimation (Chessboard)', img)
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break

    video.release()
    cv.destroyAllWindows()
