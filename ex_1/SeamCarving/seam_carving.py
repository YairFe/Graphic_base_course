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
    work_image = [np.copy(image), np.copy(image)]
    num_of_seam_to_remove = [image.shape[1] - out_width, image.shape[0] - out_height]
    for dimension in range(2):
        prev_image = np.copy(work_image[dimension + 1])
        image_indices = np.indices(work_image[dimension + 1].shape[0:2])
        index_array = np.stack((image_indices[0], image_indices[1]), axis=2)
        removed_seams = []
        grayscale_image = utils.to_grayscale(work_image[dimension + 1])
        for i in range(abs(num_of_seam_to_remove[dimension])):
            E_matrix = utils.get_gradients(work_image[dimension + 1])
            CL, CV, CR = get_CL_CV_CR(grayscale_image)
            M_matrix = calculate_energy(E_matrix, CL, CV, CR, forward_implementation)
            seam_to_remove = calculate_remove_seam(M_matrix, E_matrix, CL, CV, CR, forward_implementation)
            work_image[dimension + 1] = remove_seam_from_matrix(work_image[dimension + 1],seam_to_remove)
            grayscale_image = remove_seam_from_matrix(grayscale_image,seam_to_remove)
            removed_seams.append(index_array[np.arange(index_array.shape[0]),seam_to_remove])
            index_array = remove_seam_from_matrix(index_array,seam_to_remove)
        
        all_seam_list.append(removed_seams)

        if (num_of_seam_to_remove[dimension] < 0):
            work_image[dimension + 1] = prev_image
            for seam in removed_seams:
                work_image[dimension + 1] = duplicate_seam_in_matrix(work_image[dimension + 1], seam[:,1])
            
        work_image.append(np.rot90(work_image[dimension + 1]))

    all_seam_list[0] = np.array(all_seam_list[0]).reshape(abs(num_of_seam_to_remove[0]), work_image[0].shape[0], 2)
    all_seam_list[1] = np.array(all_seam_list[1]).reshape(abs(num_of_seam_to_remove[1]), work_image[1].shape[1], 2)

    if (all_seam_list[0].size > 0):
        work_image[0][all_seam_list[0][:,:,0], all_seam_list[0][:,:,1]] = [255, 0, 0]
    if (all_seam_list[1].size > 0):
        work_image[1][all_seam_list[1][:,:,1], all_seam_list[1][:,:,0]] = [0, 0, 255]
    
    return {"resized": np.rot90(work_image[3], 2), "vertical_seams": work_image[0], "horizontal_seams": work_image[1]}
    

def duplicate_seam_in_matrix(matrix, seam):
    res = np.zeros((matrix.shape[0], matrix.shape[1] + 1, 3))
    
    to_duplicate = np.zeros((matrix.shape[0], matrix.shape[1] + 1), bool)
    to_duplicate[np.arange(matrix.shape[0]), seam] = True

    free_space = np.zeros((matrix.shape[0], matrix.shape[1] + 1), bool)
    free_space[np.arange(matrix.shape[0]), seam + 1] = True

    to_copy = np.ones((matrix.shape[0], matrix.shape[1] + 1), bool)
    to_copy[np.arange(matrix.shape[0]), seam + 1] = False
    
    res[to_copy] = matrix.reshape((matrix.shape[0] * matrix.shape[1], 3))
    res[free_space] = res[to_duplicate]
    
    return res

def calculate_remove_seam(M_matrix, E_matrix, CL, CV, CR, forward_implementation):
    seam_to_remove = np.zeros(M_matrix.shape[0], int)

    index = np.argmin(M_matrix[-1])
    seam_to_remove[-1] = index

    for line in range(M_matrix.shape[0] - 2, -1, -1):
        # M(i-1,j-1)
        left_value = np.roll(M_matrix[line, :], 1)
        left_value[0] = np.inf
        # M(i-1,j+1)
        right_value = np.roll(M_matrix[line, :], -1)
        right_value[-1] = np.inf
        
        if forward_implementation:
            left_value += CL[line + 1, :]
            right_value += CR[line + 1, :]
            
        if (M_matrix[line + 1][index] == E_matrix[line + 1][index] + left_value[index] and index > 0):
            index -= 1
        elif (M_matrix[line + 1][index] == E_matrix[line + 1][index] + right_value[index] and index < M_matrix.shape[1] - 1):
            index += 1
            
        seam_to_remove[line] = index

    return seam_to_remove

def calculate_energy(E_matrix: NDArray, CL: NDArray, CV: NDArray, CR: NDArray, forward_implementation: bool):
    """
    Calculate the matrix M in the seam carving algorithm
    """
    M_matrix = np.zeros_like(E_matrix)
    M_matrix[0, :] = E_matrix[0, :]

    for line in range(1, E_matrix.shape[0]):
        # M(i-1,j-1)
        left_value = np.roll(M_matrix[line-1, :], 1)
        left_value[0] = np.inf
        # M(i-1,j+1)
        right_value = np.roll(M_matrix[line-1, :], -1)
        right_value[-1] = np.inf
        center_value = M_matrix[line-1, :]
        if forward_implementation:
            left_value += CL[line, :]
            right_value += CR[line, :]
            center_value += CV[line, :]
        M_matrix[line, :] = E_matrix[line, :] + np.minimum.reduce([left_value, right_value, center_value])
        #print(f"left: {left_value[:10]}")
        #print(f"right: {right_value[:10]}")
        #print(f"center: {center_value[:10]}")
        #print(f"minimum: {(M_matrix[line, :] - E_matrix[line, :])[:10]}")
    return M_matrix

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
    CV_Matrix[:, 1:-1] = first_difference[:, 1:-1]
    CL_Matrix[1:, 1:] = CV_Matrix[1:, 1:] + np.abs(down_rotate - right_rotate)[1:, 1:]
    CR_Matrix[1:, :-1] = CV_Matrix[1:, :-1] + np.abs(down_rotate - left_rotate)[1:, :-1]

    return CL_Matrix, CL_Matrix, CR_Matrix


def remove_seam_from_matrix(matrix: NDArray, seam: list):
    """
    matrix: the matrix to remove the seam
    seam: list of column indexes in matrix must be of size equal to num of lines in matrix
    return: new matrix with one less column from old matrix
    """
    res = np.zeros_like(matrix)
    res = np.delete(res, -1, 1)
    
    to_copy = np.ones(matrix.shape[0:2], bool)
    to_copy[np.arange(matrix.shape[0]), seam] = False

    res = matrix[to_copy].reshape(res.shape)
    
    return res
