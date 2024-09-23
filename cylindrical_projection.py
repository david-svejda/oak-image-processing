import cv2
import numpy as np

import argparse
from pathlib import Path


def cylindrical_warp(_image, _transformation_params):
    _height, _width = _image.shape[:2]

    _row_indices, _col_indices = np.indices((_height, _width))

    # Stacked array of col, row and ones - into 3 onedimensional arrays
    _pixel_cords = np.stack([_col_indices, _row_indices, np.ones_like(_col_indices)], axis=-1).reshape(_height * _width, 3)
    # Inverse matrix to the parameter setup
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


def Convert_xy(x, y):
    global center, f

    xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]

    return xt, yt


def ProjectOntoCylinder(InitialImage, focal=1100):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    f = focal

    # Creating a blank transformed image
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)

    # Storing all coordinates of the transformed image in 2 arrays (x and y coordinates)
    AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]

#    print(ti_x[595:605], len(ti_x))
#    print(ti_y[580:620], len(ti_y))
    # Finding corresponding coordinates of the transformed image in the initial image
    ii_x, ii_y = Convert_xy(ti_x, ti_y)

    # Rounding off the coordinate values to get exact pixel values (top-left corner)
    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)
#    print(ii_tl_x[595:605], len(ii_tl_x))
#    print(ii_tl_y[580:620], len(ii_tl_y))

    # Finding transformed image points whose corresponding
    # initial image points lies inside the initial image
    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * \
                  (ii_tl_y >= 0) * (ii_tl_y <= (h-2))

#    print(GoodIndices[595:605])
    # Removing all the outside points from everywhere
    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]

    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]

    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]

    # Bilinear interpolation
    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx)       * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx)       * (dy)

    TransformedImage[ti_y, ti_x, :] = ( weight_tl[:, None] * InitialImage[ii_tl_y,     ii_tl_x,     :] ) + \
                                      ( weight_tr[:, None] * InitialImage[ii_tl_y,     ii_tl_x + 1, :] ) + \
                                      ( weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x,     :] ) + \
                                      ( weight_br[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x + 1, :] )


    # Getting x coorinate to remove black region from right and left in the transformed image
    min_x = min(ti_x)

    # Cropping out the black region from both sides (using symmetricity)
    TransformedImage = TransformedImage[:, min_x : -min_x, :]

    return TransformedImage, ti_x-min_x, ti_y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='cylindrical_warp',
        description='Cylindrical projection',
        add_help=True
    )
    parser.add_argument('file', type=argparse.FileType('r'), nargs='+')
    parser.add_argument('-f', '--focal', type=int, default=700)
    args = parser.parse_args()

    stitched = None
    for file in args.file:
        image = cv2.imread(file.name)
        height, width = image.shape[:2]
        transformation_params = np.array([[args.focal, 0, width / 2], [0, args.focal, height / 2], [0, 0, 1]])

        #projection, _, _ = ProjectOntoCylinder(image, args.focal)
        projection = cylindrical_warp(image, transformation_params)

        cv2.imwrite("output/" + Path(file.name).stem + "_out.png", projection)

        if stitched is None:
            stitched = projection
        else:
            stitched = np.hstack((stitched, projection))

    cv2.imwrite("output/output.png", stitched)