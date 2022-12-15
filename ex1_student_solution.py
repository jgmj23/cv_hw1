"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple

from numpy.linalg import svd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""

    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        """INSERT YOUR CODE HERE"""

        A = np.zeros((2 * match_p_src.shape[1], 9))
        A[::2, 0:2] = np.transpose(match_p_src)
        A[::2, 2] = 1
        A[1::2, 3:5] = np.transpose(match_p_src)
        A[1::2, 5] = 1
        A[::2, 6:8] = np.transpose(-match_p_src * match_p_dst[0, :])
        A[::2, 8] = np.transpose(-match_p_dst[0, :])
        A[1::2, 6:8] = np.transpose(-match_p_src * match_p_dst[1, :])
        A[1::2, 8] = np.transpose(-match_p_dst[1, :])

        A = A.transpose().dot(A)

        U, S, V = svd(A)
        H = V[-1, :].reshape(3, 3)
        H = H / H[-1, -1]
        return H

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        dst_img = np.zeros(dst_image_shape, dtype=int)

        w, h, c = src_image.shape
        for i in range(w):
            for j in range(h):
                dst_homo_coor = homography.dot(np.array([j, i, 1]))
                dst_homo_coor /= dst_homo_coor[2]
                dst_homo_coor = np.round(dst_homo_coor).astype(int)
                u, v = dst_homo_coor[0], dst_homo_coor[1]
                if (u < dst_image_shape[1] and u >= 0) and (v < dst_image_shape[0] and v >= 0):
                    dst_img[v, u, :] = src_image[i, j, :]

        return dst_img

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""

        h, w, color = src_image.shape
        im_dst = np.zeros(dst_image_shape, dtype=int)
        x, y = np.mgrid[:h, :w]

        coords_hom = np.vstack((y.reshape(1, -1), x.reshape(1, -1), np.ones((1, h * w))))
        fwd = homography @ coords_hom
        fwd = np.vstack((fwd[0, :] / fwd[2, :], fwd[1, :] / fwd[2, :]))

        src_idxs = np.round(fwd).astype(int)
        valid_idx = np.where((src_idxs[0, :] < w) & (src_idxs[1, :] < h) &
                             (src_idxs[0, :] >= 0) & (src_idxs[1, :] >= 0))
        # print(valid_idx)
        coords_hom = coords_hom[:2, :]
        dst_idx_valid = np.squeeze(coords_hom[:, valid_idx]).astype(int)[::-1]
        src_idx_valid = np.squeeze(src_idxs[:, valid_idx]).astype(int)

        im_dst[src_idx_valid[1, :], src_idx_valid[0, :], :] = src_image[dst_idx_valid[0, :],
                                                              dst_idx_valid[1, :], :]

        return im_dst

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square dist of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""

        h_srcom = np.vstack([match_p_src, np.ones(match_p_src.shape[1])])
        dst_hom = np.vstack([match_p_dst, np.ones(match_p_dst.shape[1])])
        estimated = np.matmul(homography, h_srcom)
        estimated /= estimated[2, :]
        dist = estimated - dst_hom
        norms = np.array([np.linalg.norm(dist[:2, i]) for i in range(dist.shape[1])])
        fit_points = (norms < max_err)
        fit_percent = np.mean(fit_points)
        dist_mse = np.mean(norms[fit_points])
        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        h_srcom = np.vstack([match_p_src, np.ones(match_p_src.shape[1])])
        dst_hom = np.vstack([match_p_dst, np.ones(match_p_dst.shape[1])])
        estimated = np.matmul(homography, h_srcom)
        estimated /= estimated[2, :]
        dist = estimated - dst_hom
        norms = np.array([np.linalg.norm(dist[:2, i]) for i in range(dist.shape[1])])
        fit_points = (norms < max_err)
        mp_src_meets_model = match_p_src[:, fit_points]
        mp_dst_meets_model = match_p_dst[:, fit_points]

        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1

        max_fit_percent = 0
        dist_mse = np.inf

        for i in range(k):
            sampled_idx = np.random.randint(match_p_src.shape[1], size=n)
            H_test = self.compute_homography_naive(match_p_src[:, sampled_idx], match_p_dst[:, sampled_idx])
            inliers_percent, mse_temp = self.test_homography(H_test, match_p_src, match_p_dst, t)

            if inliers_percent > max_fit_percent:

                max_fit_percent = inliers_percent
                inliers_src, inliers_dst = self.meet_the_model_points(H_test, match_p_src, match_p_dst, t)
                H_test = self.compute_homography_naive(inliers_src, inliers_dst)
                inliers_percent, mse_new = self.test_homography(H_test, match_p_src, match_p_dst, t)

                if mse_new < dist_mse:
                    max_fit_percent = inliers_percent
                    print("This homography is better. MSE old = {} | MSE new = {}. Saving Model.".format(dist_mse,
                                                                                                         mse_new))
                    print("This homography achieves inliers_percent = {}.".format(max_fit_percent))
                    homography = H_test
                    dist_mse = mse_new

        print("Maximum inliers_percent achieved in {} iterations = {}.".format(k, max_fit_percent))
        if max_fit_percent > d:
            print("Maximum inliers_percent meets design requirements (d>0.5)")
        else:
            print("Maximum inliers_percent DOES NOT meet design requirements (d>0.5)")

        return homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """
        backward_warp = np.zeros(dst_image_shape, dtype=int)
        x_dst, y_dst = np.mgrid[:dst_image_shape[1], :dst_image_shape[0]]

        dst_idx = np.vstack(
            (x_dst.reshape(1, -1), y_dst.reshape(1, -1), np.ones((1, dst_image_shape[1] * dst_image_shape[0]))))
        bckwd = backward_projective_homography @ dst_idx

        bckwd /= bckwd[2]
        bckwd = np.vstack((bckwd[1], bckwd[0]))

        valid_idx = np.where((bckwd[0, :] < src_image.shape[0]) & (bckwd[1, :] < src_image.shape[1]) &
                             (bckwd[0, :] >= 0) & (bckwd[1, :] >= 0))
        bckwd = np.array(bckwd[:, valid_idx[0]]).T
        dst_idx = np.array(dst_idx[:2, valid_idx[0]], dtype=int).T

        x_src, y_src = np.mgrid[:src_image.shape[1], :src_image.shape[0]]
        src_idx = np.hstack((y_src.reshape(-1, 1, order='F'), x_src.reshape(-1, 1, order='F')))

        channels = src_image[:, :, :].reshape(-1, 1, 3)
        grid_c = griddata(src_idx, channels, bckwd, method='cubic')
        interpolations = np.squeeze(grid_c)
        backward_warp[dst_idx[:, 1], dst_idx[:, 0], :] = interpolations
        backward_warp = np.clip(backward_warp, 0, 255).astype(np.uint8)

        return backward_warp

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        """INSERT YOUR CODE HERE"""
        translation = np.eye(3)
        translation[0, 2] -= pad_left
        translation[1, 2] -= pad_up
        final_homography = backward_homography @ translation
        return final_homography


    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """

        homography=self.compute_homography(match_p_src,match_p_dst, inliers_percent, max_err)
        panorama_shape=np.array([0,0,3])

        bkwd_homography=np.linalg.inv(homography)
        bkwd_homography/=bkwd_homography[-1,-1]

        panorama_shape[0], panorama_shape[1], padding=self.find_panorama_shape(src_image,dst_image,homography)
        img_panorama = np.zeros(panorama_shape, dtype=int)
        img_panorama[padding.pad_up:padding.pad_up + dst_image.shape[0],
        padding.pad_left:padding.pad_left + dst_image.shape[1], :] = dst_image

        final_homography=self.add_translation_to_backward_homography(bkwd_homography,padding.pad_left, padding.pad_up)
        back_mapped=self.compute_backward_mapping(final_homography,src_image,panorama_shape)
        img_panorama = np.where(img_panorama > 0, img_panorama, back_mapped)

        return np.clip(img_panorama, 0, 255).astype(np.uint8)

