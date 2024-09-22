import math
import cv2
import numpy as np


def cylindrical_warp(_image, _transformation_params):
    _height, _width = _image.shape[:2]

    _row_indices, _col_indices = np.indices((_height, _width))

    _pixel_cords = np.stack([_col_indices, _row_indices, np.ones_like(_col_indices)], axis=-1).reshape(_height * _width,
                                                                                                       3)
    _transformation_params_inv = np.linalg.inv(_transformation_params)

    _pixel_cords = _transformation_params_inv.dot(_pixel_cords.T).T

    _cylindrical_cords = np.stack([np.sin(_pixel_cords[:, 0]), _pixel_cords[:, 1], np.cos(_pixel_cords[:, 0])],
                                  axis=-1).reshape(_width * _height, 3)
    _warp_coords = _transformation_params.dot(_cylindrical_cords.T).T
    _warp_coords = _warp_coords[:, :-1] / _warp_coords[:, [-1]]

    _warp_coords[(_warp_coords[:, 0] < 0) | (_warp_coords[:, 0] >= _width) | (_warp_coords[:, 1] < 0) | (
                _warp_coords[:, 1] >= _height)] = -1
    _warp_coords = _warp_coords.reshape(_height, _width, -1)

    _image_rgba = cv2.cvtColor(_image, cv2.COLOR_BGR2BGRA)
    _image_transformed = cv2.remap(src=_image_rgba,
                                   map1=_warp_coords[:, :, 0].astype(np.float32),
                                   map2=_warp_coords[:, :, 1].astype(np.float32),
                                   interpolation=cv2.INTER_AREA,
                                   borderMode=cv2.BORDER_TRANSPARENT)

    return _image_transformed


if __name__ == '__main__':
    image = cv2.imread("img.png")
    image1 = cv2.imread("img_1.png")
    image2 = cv2.imread("img_2.png")
    image3 = cv2.imread("img_3.png")

    images = [image, image1, image2, image3]

    camera_params = {
        'north': {
            'DIM': (1920, 1200),
            'K': np.array([[1148.29, 0, 992.15], [0, 1147.89, 596.26], [0, 0, 1]]),
            'D': np.array([-0.401450741, -1.36719791, 0.0000545706579, -0.0000863217663, -0.0767358005])
        },
        'east': {
            'DIM': (1920, 1200),
            'K': np.array([[1150.69, 0, 934.89], [0, 1149.70, 608.88], [0, 0, 1]]),
            'D': np.array([0.0909060544, 6.90742132, -0.000404100536, -0.00102756223, -2.77849125])
        },
        'south': {
            'DIM': (1920, 1200),
            'K': np.array([[1148.40, 0, 980.07], [0, 1147.72, 581.11], [0, 0, 1]]),
            'D': np.array([0.666609584, 8.99454152, -0.000301715763, 0.000783849089, -3.88925695])
        },
        'west': {
            'DIM': (1920, 1200),
            'K': np.array([[1147.72, 0, 947.77], [0, 1147.97, 610.60], [0, 0, 1]]),
            'D': np.array([-0.824406673, -21.3408189, -0.000582986314, 0.000503111926, 72.9079469])
        }
    }


    S = camera_params['north']['K'][0][0]

    height, width = image.shape[:2]
    transformation_params = np.array([[S, 0, width / 2],
                                      [0, S, height / 2],
                                      [0, 0, 1]])

    t_images = []
    for i in range(len(images)):
        warp = cylindrical_warp(images[i], transformation_params)
        t_images.append(warp)

    # Execute cylindrical warp
    for i in range(len(t_images)):
        cv2.imshow(f"im{i}", t_images[i])
    cv2.waitKey()

    stitched = np.hstack([t_images[3], t_images[0], t_images[1], t_images[2]])

    cv2.imwrite("output0.png", stitched)
