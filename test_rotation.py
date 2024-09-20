import math
import cv2
import numpy as np


def cylindrical_warp(_image, _transformation_params):
    _height, _width = _image.shape[:2]

    _row_indices, _col_indices = np.indices((_height, _width))

    _pixel_cords = np.stack([_col_indices, _row_indices, np.ones_like(_col_indices)], axis=-1).reshape(_height * _width, 3)
    _transformation_params_inv = np.linalg.inv(_transformation_params)

    _pixel_cords = _transformation_params_inv.dot(_pixel_cords.T).T

    _cylindrical_cords = np.stack([np.sin(_pixel_cords[:, 0]), _pixel_cords[:, 1], np.cos(_pixel_cords[:, 0])], axis=-1).reshape(_width * _height, 3)
    _warp_coords = _transformation_params.dot(_cylindrical_cords.T).T
    _warp_coords = _warp_coords[:, :-1] / _warp_coords[:, [-1]]

    _warp_coords[(_warp_coords[:, 0] < 0) | (_warp_coords[:, 0] >= _width) | (_warp_coords[:, 1] < 0) | (_warp_coords[:, 1] >= _height)] = -1
    _warp_coords = _warp_coords.reshape(_height, _width, -1)
    
    _image_rgba = cv2.cvtColor(_image, cv2.COLOR_BGR2BGRA)
    _image_transformed = cv2.remap(src=_image_rgba,
                                   map1=_warp_coords[:, :, 0].astype(np.float32),
                                   map2=_warp_coords[:, :, 1].astype(np.float32),
                                   interpolation=cv2.INTER_AREA,
                                   borderMode=cv2.BORDER_TRANSPARENT)

    return _image_transformed


if __name__ == '__main__':
    image = cv2.imread("cam_undist_4.png")

    height, width = image.shape[:2]
    transformation_params = np.array([[400, 0, width / 2], [0, 400, height / 2], [0, 0, 1]])

    # Execute cylindrical warp
    cv2.imwrite("output/output01.png", cylindrical_warp(image, transformation_params))