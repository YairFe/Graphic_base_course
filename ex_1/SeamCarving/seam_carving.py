from typing import Dict, Any
import utils
import numpy as np

NDArray = Any


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    all_seam_list = [] # index 0 all column seams to remove, index 1 all row seam to remove
    work_image = np.copy(image)
    num_of_seam_to_remove = [abs(out_width - image.shape[1]), abs(out_height - image.shape[0])]
    for dimension in range(2):
        image_indices = np.indices(image.shape)
        index_array = np.stack((image_indices[0], image_indices[1]), axis=2)
        removed_seams = []
        grayscale_image = utils.to_grayscale(image)
        for i in range(num_of_seam_to_remove[dimension]):
            E_matrix = utils.get_gradients(work_image)
            CL, CV, CR = get_CL_CV_CR(grayscale_image)
            M_matrix = calculate_energy(E_matrix, CL, CV, CR, forward_implementation)
            # seam_to_remove = calculate_remove_seam(M_matrix, CL, CV, CR, forward_implemtation)
            # work_image = remove_seam_from_matrix(work_image,seam_to_remove)
            # grayscale_image = remove_seam_from_matrix(grayscale_image,seam_to_remove)
            # removed_seam.append(index_array[np.arange(index_array.shape[0]),seam_to_remove])
            # index_array = remove_seam_from_matrix(index_array,seam_to_remove)

        all_seam_list.append(removed_seams)
        work_image = np.rot90(work_image)

    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}
    return {"energy": E_matrix, "grayscale": grayscale_image}

def calculate_energy(E_matrix: NDArray, CL: NDArray, CV: NDArray, CR: NDArray, forward_implementation: bool):
    """
    Calculate the martix M in the seam carving algorithm
    """
    M_martix = np.zeros_like(E_matrix)
    M_martix[0, :] = E_matrix[0, :]

    for line in range(1, E_matrix.shape[0]):
        # M(i-1,j-1)
        left_value = np.roll(M_martix[line-1, :], 1)
        left_value[0] = np.inf
        # M(i-1,j+1)
        right_value = np.roll(M_martix[line-1, :], -1)
        right_value[-1] = np.inf
        center_value = M_martix[line-1, :]
        if forward_implementation:
            left_value += CL[line, :]
            right_value += CR[line, :]
            center_value += CV[line, :]
        M_martix[line, :] = E_matrix[line, :] + np.minimum.reduce([left_value, right_value, center_value])
        print(f"left: {left_value[:10]}")
        print(f"right: {right_value[:10]}")
        print(f"center: {center_value[:10]}")
        print(f"minimum: {(M_martix[line, :] - E_matrix[line, :])[:10]}")
    return M_martix

def get_CL_CV_CR(grayscale_image: NDArray):
    CL_Matrix = np.full_like(grayscale_image, 255.0)
    CV_Matrix = np.full_like(grayscale_image, 255.0)
    CR_Matrix = np.full_like(grayscale_image, 255.0)
    # image_grayscale(i,j+1)
    left_rotate = np.roll(grayscale_image, -1, axis=1)
    # image_grayscale(i,j-1)
    right_rotate = np.roll(grayscale_image, 1, axis=1)
    # image_grayscale(i-1,j)
    down_rotate = np.roll(grayscale_image, shift=1, axis=0)
    first_difference = np.abs(left_rotate - right_rotate)
    # all the values in
    CV_Matrix[1:-1, 1:-1] = first_difference[1:-1, 1:-1]
    CL_Matrix[1:-1, 1:-1] = first_difference[1:-1, 1:-1] + np.abs(down_rotate - right_rotate)[1:-1, 1:-1]
    CR_Matrix[1:-1, 1:-1] = first_difference[1:-1, 1:-1] + np.abs(down_rotate - left_rotate)[1:-1, 1:-1]

    return CL_Matrix, CL_Matrix, CR_Matrix


def remove_seam_from_matrix(matrix: NDArray, seam: list):
    """
    matrix: the matrix to remove the seam
    seam: list of column indexes in matrix must be of size equal to num of lines in matrix
    return: new matrix with one less column from old matrix
    """
    tmp = np.arange(matrix.shape[0])
    matrix[tmp, -1], matrix[tmp, seam] = matrix[tmp, seam], matrix[tmp, -1]
    return np.delete(matrix, -1, axis=1)
