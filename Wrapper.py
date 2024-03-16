#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
from skimage import transform

# matplotlib.use("TKAgg")
from sklearn.cluster import KMeans
from scipy.ndimage import convolve


class CustomUtils:
    def __init__(self):
        self.description = (
            "Custom Utilities"  # Mostly Rotation and Convolution Utilities
        )

    def convolve(self, image, filter):
        """
        Convolve the image with the filter.

        Args:
            image (numpy.ndarray): The input image.
            filter (numpy.ndarray): The filter to convolve with.

        Returns:
            numpy.ndarray: The convolved image.
        """
        image_row, image_col = image.shape
        filter_row, filter_col = filter.shape
        output = np.zeros(image.shape)
        padded_image = np.pad(image, (filter_row // 2, filter_col // 2), "constant")
        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(
                    filter
                    * padded_image[row : row + filter_row, col : col + filter_col]
                )
        return output


class FileHandler:
    def __init__(self, basepath="Results/"):
        """
        Initializes a FileHandler object.

        Parameters:
        - basepath (str): The base path where the output files will be saved. Default is "Results/".
        """
        self.base_path = basepath

    def plot_images(self, fig_size, filters, x_len, y_len, name):
        """
        Plots a grid of images and saves it as a file.

        Parameters:
        - fig_size (tuple): The size of the figure (width, height).
        - filters (list): A list of images to be plotted.
        - x_len (int): The number of columns in the grid.
        - y_len (int): The number of rows in the grid.
        - name (str): The name of the output file.
        """
        fig = plt.figure(figsize=fig_size)
        length = len(filters)
        for idx in np.arange(length):
            ax = fig.add_subplot(y_len, x_len, idx + 1, xticks=[], yticks=[])
            plt.imshow(filters[idx], cmap="gray")
        plt.axis("off")
        plt.savefig(name, bbox_inches="tight", pad_inches=0.3)
        plt.close()

    # ToDO: Fix this Issue
    def plot_images_cv2(self, fig_size, filters, x_len, y_len, name):
        """
        Plots a grid of images and saves it as a file.

        Parameters:
        - fig_size (tuple): The size of the figure (width, height).
        - filters (list): A list of images to be plotted.
        - x_len (int): The number of columns in the grid.
        - y_len (int): The number of rows in the grid.
        - name (str): The name of the output file.
        """
        # Create a blank image
        total_width = x_len * fig_size[0]
        total_height = y_len * fig_size[1]
        output_image = np.zeros((total_height, total_width), dtype=np.uint8)

        length = len(filters)
        for idx in np.arange(length):
            # Calculate the position of the current filter in the grid
            row = idx // x_len
            col = idx % x_len

            # Calculate the region where the current filter will be placed
            y_start = row * fig_size[1]
            y_end = y_start + fig_size[1]
            x_start = col * fig_size[0]
            x_end = x_start + fig_size[0]

            # Resize the filter image to match the specified size
            filter_image = cv2.resize(filters[idx], (fig_size[0], fig_size[1]))

            # Place the filter image in the output image
            output_image[y_start:y_end, x_start:x_end] = filter_image

        # Save the resulting image
        cv2.imwrite(name, output_image)

    def check_folder_exists(self, folder_path):
        """
        Checks if a folder exists at the specified path, and creates it if it doesn't exist.

        Parameters:
        - folder_path (str): The path of the folder to be checked.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def write_output(self, image, type_of_filter, name):
        """
        Writes the output image to the base path.

        Parameters:
        - image: The image to be written.
        - type_of_filter (str): The type of filter applied to the image.
        - name (str): The name of the output file.
        """
        self.check_folder_exists(self.base_path + type_of_filter)
        cv2.imwrite(
            self.base_path + type_of_filter + name,
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )


class FilterBank:
    def __init__(self):
        """
        Initializes the Wrapper object with default values for various parameters.

        Parameters:
        - description (str): The description of the Wrapper object.
        - kernel_size (int): The size of the kernel for filter banks.
        - dog_sigma_gaussian (list): The sigma values for Difference of Gaussian (DoG) filters.
        - dog_num_orientations (int): The number of orientations for DoG filters.
        - sobel_filter (numpy.ndarray): The Sobel filter.
        - laplacian_filter (numpy.ndarray): The Laplacian filter.
        - cluster_count_texton (int): The number of clusters for texton.
        - cluster_count_color (int): The number of clusters for color.
        - cluster_count_brightness (int): The number of clusters for brightness.
        - lm_num_orientations (int): The number of orientations for local maxima.
        - lm_sigma_gaussian_list_small (list): The sigma values for local maxima (small scales).
        - lm_sigma_gaussian_list_big (list): The sigma values for local maxima (big scales).
        - lmsize (int): The size of the local maxima.
        - lm_elongation_factor (int): The elongation factor for local maxima.
        - gabor_num_orientations (int): The number of orientations for Gabor filters.
        - gabor_sigma_gaussian (list): The sigma values for Gabor filters.
        - gb_Lambda (int): The Lambda value for Gabor filters.
        - gb_psi (int): The psi value for Gabor filters.
        - gb_gamma (int): The gamma value for Gabor filters.
        - half_disk_radius_list (list): The radius values for half disk filters.
        - half_disk_num_orientations (int): The number of orientations for half disk filters.
        """
        self.description = "Filter Banks"
        self.kernel_size = 49  # 7x7
        self.dog_sigma_gaussian = [3, 5]  # 2 Gaussian Scales
        self.dog_num_orientations = 16
        self.sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.cluster_count_texton = 64
        self.cluster_count_color = 16
        self.cluster_count_brightness = 16
        self.lm_num_orientations = 6
        self.lm_sigma_gaussian_list_small = [1, np.sqrt(2), 2, np.sqrt(8)]
        self.lm_sigma_gaussian_list_big = [np.sqrt(2), 2, np.sqrt(8), 4]
        self.lmsize = 49
        self.lm_elongation_factor = 3
        self.gabor_num_orientations = 8
        self.gabor_sigma_gaussian = [3, 5, 7, 9, 11]  # 5 Gaussian Scales
        self.gb_Lambda = 4
        self.gb_psi = 1
        self.gb_gamma = 1
        self.half_disk_radius_list = [5, 10, 15]
        self.half_disk_num_orientations = 16

    def rotate_filter(self, filter, theta):
        """
        Rotate the given filter by the specified angle.

        Parameters:
        filter (numpy.ndarray): The filter to be rotated.
        theta (float): The angle of rotation in degrees.

        Returns:
        numpy.ndarray: The rotated filter.
        """
        rotated_filter = transform.rotate(filter, theta, resize=False)
        return rotated_filter

    def gaussian1d(self, x, sigma, derivative_order):
        """
        Generate a Gaussian filter for edge detection.

        Parameters:
        x (float): The input value.
        sigma (float): The standard deviation of the Gaussian distribution.
        derivative_order (int): The order of the derivative.

        Returns:
        float: The value of the Gaussian filter.
        """
        g_filter_out = np.exp(-((x**2) / (2.0 * sigma**2))) / (
            np.sqrt(2 * np.pi) * sigma
        )
        if derivative_order == 1:
            g_filter_out = -x * g_filter_out / (sigma**2)
        elif derivative_order == 2:
            g_filter_out = (x**2 - sigma**2) * g_filter_out / (sigma**4)
        return g_filter_out

    def return_meshgrid(self, size):
        """
        Returns a meshgrid of coordinates.

        Parameters:
        size (int): The size of the meshgrid.

        Returns:
        x (ndarray): The x-coordinates of the meshgrid.
        y (ndarray): The y-coordinates of the meshgrid.
        """
        interval = size // 2
        [x, y] = np.meshgrid(
            np.linspace(-interval, interval, size),
            np.linspace(-interval, interval, size),
        )
        return x, y

    def gaussian_filter2d(self, sigma, d_order_x, d_order_y):
        """
        Generate a 2D Gaussian filter for edge detection.

        Parameters:
        sigma (float): Standard deviation of the Gaussian distribution.
        d_order_x (int): Order of derivative in x-direction.
        d_order_y (int): Order of derivative in y-direction.

        Returns:
        numpy.ndarray: 2D Gaussian filter.
        """
        [x, y] = self.return_meshgrid(self.kernel_size)
        grid = np.array([x.flatten(), y.flatten()])
        g_x = self.gaussian1d(grid[0], sigma, d_order_x)
        g_y = self.gaussian1d(grid[1], 3 * sigma, d_order_y)
        g_filter_out = g_x * g_y
        g_filter_out = g_filter_out.reshape(self.kernel_size, self.kernel_size)
        return g_filter_out

    def generic_gauss_filter2d(self, size, sigma):
        """
        Generate a Gaussian filter for edge detection.

        Args:
            size (int): The size of the filter.
            sigma (float): The standard deviation of the Gaussian distribution.

        Returns:
            numpy.ndarray: The generated Gaussian filter.
        """
        [x, y] = self.return_meshgrid(size)
        g_filter_out = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) / (
            2 * np.pi * sigma**2
        )
        return g_filter_out

    def laplacian_of_gaussian_filter(self, size, sigma):
        """
        Generate a Laplacian of Gaussian filter for edge detection.

        Parameters:
        - size: The size of the filter.
        - sigma: The standard deviation of the Gaussian filter.

        Returns:
        - laplacian_of_gaussian_filter: The Laplacian of Gaussian filter.
        """
        laplacian_of_gaussian_filter = convolve(
            self.generic_gauss_filter2d(size, sigma), self.laplacian_filter
        )
        return laplacian_of_gaussian_filter

    def gabor_filter(self, sigma, theta, Lambda, psi, gamma, size):
        """Generate a Gabor filter.

        Args:
            sigma (float): Standard deviation of the Gaussian envelope.
            theta (float): Orientation of the Gabor filter in radians.
            Lambda (float): Wavelength of the sinusoidal factor.
            psi (float): Phase offset of the sinusoidal factor.
            gamma (float): Spatial aspect ratio.
            size (int): Size of the Gabor filter.

        Returns:
            numpy.ndarray: Gabor filter.

        """
        [x, y] = self.return_meshgrid(size)
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gabor_filter_out = np.exp(
            -((x_theta**2 + gamma**2 * y_theta**2) / (2.0 * sigma**2))
        ) * np.cos(2 * np.pi * x_theta / Lambda + psi)
        return gabor_filter_out

    def predict_map_cluster(self, map, cluster_count):
        """Cluster similar responses together.

        Args:
            map (numpy.ndarray): The input map.
            cluster_count (int): The number of clusters.

        Returns:
            numpy.ndarray: The predicted cluster labels.

        """
        map_shape = map.shape
        if len(map_shape) == 3:
            map = np.reshape(map, ((map_shape[0] * map_shape[1]), map_shape[2]))
        else:
            map = np.reshape(map, ((map_shape[0] * map_shape[1]), 1))
        kmeans = KMeans(n_clusters=cluster_count, random_state=0)
        pred = kmeans.fit_predict(map)
        pred = np.reshape(pred, (map_shape[0], map_shape[1]))
        return pred

    def create_dog_filter_bank(self):
        """Create a filter bank with oriented Derivative of Gaussian (DoG) filters.

        Returns:
            dog_filter_bank (list): List of DoG filters.
        """
        dog_filter_bank = []
        count = 0
        for scale in range(len(self.dog_sigma_gaussian)):
            gaussian = self.generic_gauss_filter2d(
                self.kernel_size, self.dog_sigma_gaussian[scale]
            )
            sobel = self.sobel_filter
            dog = convolve(gaussian, sobel)
            for orientation in np.linspace(
                0, 360, self.dog_num_orientations, endpoint=False
            ):
                dog = self.rotate_filter(dog, orientation)
                dog_filter_bank.append(dog)
                count += 1
        return dog_filter_bank

    def create_lm_filter_bank(self, type="small"):
        """
        Creates a filter bank with Leung-Malik (LM) filters.

        Args:
            type (str): The type of LM filter bank to create. Options are "small" (default) or "big".

        Returns:
            list: The LM filter bank, which is a list of filter kernels.

        Notes:
            - The filter bank consists of 48 filters.
            - For the "small" type, the first three filters in the sigma list are used for first and second order derivatives.
            - For the "big" type, the sigma list is different.
            - The first order derivative filters are Gaussian filters with 6 orientations and 3 scales.
            - The second order derivative filters are Gaussian filters with 6 orientations and 3 scales.
            - The Laplacian of Gaussian filters are used with 8 filters, where sigma is multiplied by the elongation factor.
            - Gaussian filters occur at 4 scales.
        """
        lm_filter_bank = []
        if type == "small":
            lm_sigma_gaussian_list = self.lm_sigma_gaussian_list_small
        else:
            lm_sigma_gaussian_list = self.lm_sigma_gaussian_list_big

        for i in range(len(lm_sigma_gaussian_list)):
            temp_sigma = lm_sigma_gaussian_list[i]
            if i < 3:
                # Multiplying by Elongation Factor
                gaussian_derivative = self.gaussian_filter2d(
                    temp_sigma, d_order_x=1, d_order_y=0
                )
                gaussian_second_derivative = self.gaussian_filter2d(
                    temp_sigma, d_order_x=2, d_order_y=0
                )
                for orientation in np.linspace(
                    0, 180, self.lm_num_orientations, endpoint=False
                ):
                    gaussian_derivative = self.rotate_filter(
                        gaussian_derivative, orientation
                    )
                    gaussian_second_derivative = self.rotate_filter(
                        gaussian_second_derivative, orientation
                    )
                    lm_filter_bank.append(gaussian_derivative)
                    lm_filter_bank.append(gaussian_second_derivative)

        for i in range(len(lm_sigma_gaussian_list)):
            temp_sigma = lm_sigma_gaussian_list[i]
            laplacian_of_gaussian = self.laplacian_of_gaussian_filter(
                self.kernel_size, temp_sigma
            )
            lm_filter_bank.append(laplacian_of_gaussian)

        for i in range(len(lm_sigma_gaussian_list)):
            temp_sigma = lm_sigma_gaussian_list[i]
            laplacian_of_gaussian_elong = self.laplacian_of_gaussian_filter(
                self.kernel_size, temp_sigma * self.lm_elongation_factor
            )
            lm_filter_bank.append(laplacian_of_gaussian_elong)

        for i in range(len(lm_sigma_gaussian_list)):
            temp_sigma = lm_sigma_gaussian_list[i]
            gaussian = self.generic_gauss_filter2d(self.kernel_size, temp_sigma)
            lm_filter_bank.append(gaussian)
        return lm_filter_bank

    def create_gabor_filter_bank(self):
        """Create a filter bank with Gabor filters.

        Returns:
            list: A list of Gabor filters.
        """
        gabor_filter_bank = []
        for scale in range(len(self.gabor_sigma_gaussian)):
            for orientation in np.linspace(
                45, 225, self.gabor_num_orientations, endpoint=False
            ):
                gabor = self.gabor_filter(
                    self.gabor_sigma_gaussian[scale],
                    orientation,
                    self.gb_Lambda,
                    self.gb_psi,
                    self.gb_gamma,
                    self.kernel_size,
                )
                gabor_filter_bank.append(gabor)
        return gabor_filter_bank

    def create_half_disk_filter_bank(self):
        """Create a filter bank with half disc filters.

        Returns:
            list: A list of half disc filters.
        """
        half_disc_filter_bank = []
        orientations = np.linspace(
            0, 360, self.half_disk_num_orientations, endpoint=False
        )
        for radius in self.half_disk_radius_list:
            size = 2 * radius + 1
            filter = np.zeros((size, size))
            for i in range(radius):
                for j in range(size):
                    cond_1 = (i - radius) ** 2 + (j - radius) ** 2
                    if cond_1 <= radius**2:
                        filter[i, j] = 1

            for i in range(0, len(orientations)):
                mask = self.rotate_filter(filter, orientations[i])
                mask[mask <= 0.5] = 0
                mask[mask > 0.5] = 1
                half_disc_filter_bank.append(mask)

        return half_disc_filter_bank

    def texton_ind(self, image, filter_bank):
        """
        Computes the texton index map for the given image using the provided filter bank.

        Parameters:
        image (numpy.ndarray): The input image.
        filter_bank (list): The list of filters to be applied.

        Returns:
        numpy.ndarray: The texton index map.
        """
        tex_map = np.array(image)
        for i in range(len(filter_bank)):
            filter = np.array(filter_bank[i])
            filter_map = convolve(image, filter)
            tex_map = np.dstack((tex_map, filter_map))
        return tex_map

    def get_texton_map(self, image, dog, lm, gabor):
        """
        Generates a texton map by combining the texton indices obtained from different filters.

        Parameters:
        image (numpy.ndarray): The input image.
        dog (numpy.ndarray): The texton indices obtained from Difference of Gaussians filter.
        lm (numpy.ndarray): The texton indices obtained from Local Maxima filter.
        gabor (numpy.ndarray): The texton indices obtained from Gabor filter.

        Returns:
        numpy.ndarray: The texton map obtained by stacking the texton indices from different filters.
        """
        tex_map_dog = self.texton_ind(image, dog)
        tex_map_lm = self.texton_ind(image, lm)
        tex_map_gabor = self.texton_ind(image, gabor)
        tex_map = np.dstack((tex_map_dog, tex_map_lm, tex_map_gabor))
        return tex_map

    def get_chi_square_dist(self, map, bins, mask, inverse_mask):
        """
        Calculate the chi square distance between two histograms.

        Parameters:
        map (numpy.ndarray): The input histogram.
        bins (int): The number of bins in the histogram.
        mask (numpy.ndarray): The half disk mask.
        inverse_mask (numpy.ndarray): The complementary half disk mask.

        Returns:
        numpy.ndarray: The chi square distance between the two histograms.
        """
        chi_square_dist = map * 0
        for i in range(bins):
            tmp = np.zeros_like(map)
            tmp[map == i] = 1
            g_i = cv2.filter2D(tmp, -1, mask)
            h_i = cv2.filter2D(tmp, -1, inverse_mask)

            temp_chi_distance = ((g_i - h_i) ** 2) / (g_i + h_i + 0.005)
            chi_square_dist = chi_square_dist + temp_chi_distance
        chi_square_dist = chi_square_dist / 2
        return chi_square_dist

    def get_gradients(self, map, half_disk_filter_bank, bins):
        """
        Calculate the gradients of the given map using the half disk filter bank.

        Parameters:
        map (numpy.ndarray): The input map.
        half_disk_filter_bank (list): The list of half disk filters.
        bins (int): The number of bins for histogram calculation.

        Returns:
        numpy.ndarray: The gradients of the map.
        """
        grad = np.array(map)
        for i in range(0, len(half_disk_filter_bank), 2):
            chi_square_dist = self.get_chi_square_dist(
                map, bins, half_disk_filter_bank[i], half_disk_filter_bank[i + 1]
            )
            grad = np.dstack((grad, chi_square_dist))
        gradients = np.mean(grad, axis=2)
        return gradients


def main():
    # Initialize the classes
    custom_utils = CustomUtils()
    file_handler = FileHandler()
    filter_bank = FilterBank()

    delete_prev_run = input("Do you want to delete previous results (y/n):")
    if delete_prev_run == "y" or delete_prev_run == "Y":
        if os.path.exists("Results/"):
            shutil.rmtree("Results/")

    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """

    dog_filter_bank = filter_bank.create_dog_filter_bank()
    print("Size of DoG filter bank: ", len(dog_filter_bank))

    file_handler.check_folder_exists("Results/DoG/")
    file_handler.plot_images(
        (16, 2), dog_filter_bank, x_len=16, y_len=2, name="Results/DoG/DoG_filters.png"
    )

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """

    lm_filter_bank = filter_bank.create_lm_filter_bank(type="small")
    print("Size of LM filter bank - SMALL : ", len(lm_filter_bank))

    file_handler.check_folder_exists("Results/LMF/")
    file_handler.plot_images(
        (12, 4),
        lm_filter_bank,
        x_len=12,
        y_len=4,
        name="Results/LMF/LMF_filters_small.png",
    )

    lm_filter_bank_large = filter_bank.create_lm_filter_bank(type="large")
    print("Size of LM filter bank - LARGE: ", len(lm_filter_bank_large))

    file_handler.check_folder_exists("Results/LMF/")
    file_handler.plot_images(
        (12, 4),
        lm_filter_bank_large,
        x_len=12,
        y_len=4,
        name="Results/LMF/LMF_filters_large.png",
    )

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """

    gabor_filter_bank = filter_bank.create_gabor_filter_bank()
    print("Size of Gabor filter bank: ", len(gabor_filter_bank))

    file_handler.check_folder_exists("Results/Gabor/")
    file_handler.plot_images(
        (8, 5),
        gabor_filter_bank,
        x_len=8,
        y_len=5,
        name="Results/Gabor/Gabor_filters.png",
    )

    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    half_disk_mask_list = filter_bank.create_half_disk_filter_bank()
    print("Size of Half disk filter bank: ", len(half_disk_mask_list))

    file_handler.check_folder_exists("Results/HalfDisk/")
    file_handler.plot_images(
        (8, 6),
        half_disk_mask_list,
        x_len=8,
        y_len=6,
        name="Results/HalfDisk/HalfDisk_filters.png",
    )

    imagefiles_base_path = "../BSDS500/Images/"
    imagefiles = os.listdir(imagefiles_base_path)
    imagefiles.sort()

    imagefiles = imagefiles  # [0:1]

    for imagefile in imagefiles:
        # Turn off axis
        plt.axis("off")
        print("====" * 5)
        print("Current Image file in Progress: ", imagefile)
        # Read the image
        image_org = cv2.imread(imagefiles_base_path + imagefile)
        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        img_size = image.shape
        """
        Generate Texton Map
        Filter image using oriented gaussian filter bank
        """
        texton_map = filter_bank.get_texton_map(
            image, dog_filter_bank, lm_filter_bank, gabor_filter_bank
        )

        """
        Generate texture ID's using K-means clustering
        Display texton map and save image as TextonMap_ImageName.png,
        use command "cv2.imwrite('...)"
        """
        total_filters = texton_map.shape[2]
        print("Total Number of filters in Texton Map : ", total_filters)
        cluster_count = 64
        texton_map_pred = filter_bank.predict_map_cluster(texton_map, cluster_count)

        texton_map_pred = 3 * texton_map_pred
        cm = plt.get_cmap("gist_rainbow")
        pred = cm(texton_map_pred)
        pred1 = pred * 255
        # pred1 = pred1.astype(np.uint8)
        # pred1 = cv2.cvtColor(pred1, cv2.COLOR_RGBA2BGR)
        file_handler.write_output(pred1, "TextonMap/", "TextonMap_" + imagefile)
        plt.axis("off")
        plt.imshow(texton_map_pred)
        file_handler.check_folder_exists("Results/TextonMap/")
        plt.savefig(
            "Results/TextonMap/TextonMap_plt" + imagefile,
            bbox_inches="tight",
            pad_inches=0,
        )

        """
        Generate Texton Gradient (Tg)
        Perform Chi-square calculation on Texton Map
        Display Tg and save image as Tg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        texton_gradient_orig = filter_bank.get_gradients(
            texton_map_pred, half_disk_mask_list, cluster_count
        )
        texton_gradient1 = texton_gradient_orig
        # texton_gradient1 = texton_gradient1.astype(np.uint8)
        texton_gradient1 = cv2.convertScaleAbs(texton_gradient1)
        texton_gradient1 = cv2.cvtColor(texton_gradient1, cv2.COLOR_GRAY2BGR)
        file_handler.write_output(
            texton_gradient1,
            "TextonGradient/",
            "TextonGradient_" + imagefile,
        )
        plt.axis("off")
        plt.imshow(texton_gradient_orig, cmap="gray")
        file_handler.check_folder_exists("Results/TextonGradient/")
        plt.savefig(
            "Results/TextonGradient/TextonGradient_plt" + imagefile,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        """
        Generate Brightness Map
        Perform brightness binning  
        """
        brightness_map = np.array(image_org)
        bright_pred_orig = filter_bank.predict_map_cluster(brightness_map, 16)
        bright_pred = 16 * bright_pred_orig
        bright_pred = cm(bright_pred)
        bright_pred = bright_pred * 255
        # bright_pred = bright_pred.astype(np.uint8)
        # bright_pred = cv2.cvtColor(bright_pred, cv2.COLOR_RGBA2BGR)
        file_handler.write_output(
            bright_pred, "BrightnessMap/", "BrightnessMap_" + imagefile
        )
        plt.axis("off")
        plt.imshow(bright_pred_orig, cmap="gray")
        file_handler.check_folder_exists("Results/BrightnessMap/")
        plt.savefig(
            "Results/BrightnessMap/BrightMap_plt" + imagefile,
            bbox_inches="tight",
            pad_inches=0,
        )

        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        brightness_gradient_orig = filter_bank.get_gradients(
            bright_pred_orig, half_disk_mask_list, 16
        )
        brightness_gradient = brightness_gradient_orig
        brightness_gradient = brightness_gradient
        brightness_gradient = cv2.convertScaleAbs(brightness_gradient)
        brightness_gradient = cv2.cvtColor(brightness_gradient, cv2.COLOR_GRAY2BGR)
        # brightness_gradient = brightness_gradient.astype(np.uint8)
        file_handler.write_output(
            brightness_gradient,
            "BrightnessGradient/",
            "BrightnessGradient_" + imagefile,
        )
        plt.axis("off")
        plt.imshow(brightness_gradient_orig, cmap="gray")
        file_handler.check_folder_exists("Results/BrightnessGradient/")
        plt.savefig(
            "Results/BrightnessGradient/BrightnessGradient_plt" + imagefile,
            bbox_inches="tight",
            pad_inches=0,
        )

        """
        Generate Color Map
        Perform color binning or clustering
        """
        color_map = np.array(image_org)
        color_pred_orig = filter_bank.predict_map_cluster(color_map, 16)
        color_pred = 16 * color_pred_orig
        color_pred = cm(color_pred)
        color_pred = color_pred * 255
        # color_pred = color_pred.astype(np.uint8)
        # color_pred = cv2.cvtColor(color_pred, cv2.COLOR_RGBA2BGR)
        file_handler.write_output(color_pred, "ColorMap/", "ColorMap_" + imagefile)
        plt.axis("off")
        plt.imshow(color_pred_orig)
        file_handler.check_folder_exists("Results/ColorMap/")
        plt.savefig(
            "Results/ColorMap/ColorMap_plt" + imagefile,
            bbox_inches="tight",
            pad_inches=0,
        )

        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        color_gradient_orig = filter_bank.get_gradients(
            color_pred_orig, half_disk_mask_list, 16
        )
        color_gradient = color_gradient_orig
        # color_gradient = color_gradient.astype(np.uint8)
        color_gradient = cv2.convertScaleAbs(color_gradient)
        color_gradient = cv2.cvtColor(color_gradient, cv2.COLOR_GRAY2BGR)
        file_handler.write_output(
            color_gradient, "ColorGradient/", "ColorGradient_" + imagefile
        )
        plt.axis("off")
        plt.imshow(color_gradient_orig, cmap="gray")
        file_handler.check_folder_exists("Results/ColorGradient/")
        plt.savefig(
            "Results/ColorGradient/ColorGradient_plt" + imagefile,
            bbox_inches="tight",
            pad_inches=0,
        )

        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        """
        sobel_baseline = cv2.imread(
            "../BSDS500/SobelBaseline/" + imagefile.split(".")[0] + ".png", 0
        )

        """
        Read Canny Baseline
        use command "cv2.imread(...)"
        """
        canny_baseline = cv2.imread(
            "../BSDS500/CannyBaseline/" + imagefile.split(".")[0] + ".png", 0
        )

        print("Sobel Baseline Shape: ", sobel_baseline.shape)
        print("Canny Baseline Shape: ", canny_baseline.shape)
        print("Texton Gradient Shape: ", texton_gradient_orig.shape)
        print("Brightness Gradient Shape: ", brightness_gradient_orig.shape)
        print("Color Gradient Shape: ", color_gradient_orig.shape)

        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """
        pblite_output = (
            (1 / 3)
            * (texton_gradient_orig + brightness_gradient_orig + color_gradient_orig)
            * (0.5 * sobel_baseline + 0.5 * canny_baseline)
        )
        pblite_output1 = pblite_output
        # pblite_output1 = pblite_output.astype(np.uint8)
        file_handler.write_output(pblite_output1, "PbLite/", "PbLite_" + imagefile)
        plt.axis("off")
        plt.imshow(pblite_output, cmap="gray")
        file_handler.check_folder_exists("Results/PbLite/")
        plt.savefig(
            "Results/PbLite/PbLite_plt" + imagefile, bbox_inches="tight", pad_inches=0
        )


if __name__ == "__main__":
    main()
