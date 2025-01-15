import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import minimize
from scipy import ndimage


class DartboardTransformer:
    def __init__(self, perfect_circle=None):
        if perfect_circle is None:
            self.perfect_circle = {
                "top": [0, -1],
                "bottom": [0, 1],
                "left": [-1, 0],
                "right": [1, 0],
            }
        else:
            self.perfect_circle = perfect_circle
        self.bg_img = "./dartboard_ci.png"

    def setup_plot(self):
        """Initialize the plot window."""
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        self.img_displayed = None

    def _affine_transform(self, params, src):
        a, b, c, d, e, f = params
        transform_matrix = np.array([[a, b, c], [d, e, f], [0, 0, 1]])
        src_homogeneous = np.hstack([src, np.ones((src.shape[0], 1))])
        transformed = src_homogeneous @ transform_matrix.T
        return transformed[:, :2]

    def _objective_function(self, params, src, dst):
        transformed = self._affine_transform(params, src)
        return np.sum((transformed - dst) ** 2)

    def _compute_transformation_params(self, src_points, dst_points):
        initial_guess = [1, 0, 0, 0, 1, 0]
        result = minimize(
            self._objective_function, initial_guess, args=(src_points, dst_points)
        )
        return result.x

    def transform_point(self, point, params):
        a, b, c, d, e, f = params
        transform_matrix = np.array([[a, b, c], [d, e, f], [0, 0, 1]])
        point_homogeneous = np.array([point[0], point[1], 1])
        transformed = transform_matrix @ point_homogeneous
        return transformed[:2]

    def transform_all_points(self, coordinates, optimized_params):
        return {
            name: self.transform_point(coordinates[name], optimized_params)
            for name in coordinates
            if name not in ["top", "bottom", "left", "right"]
        }

    def _map_to_coordinates(self, xy_array):
        """Map the xy_array to coordinates."""
        xy_array_copy = xy_array.copy()
        xy_array_copy[:, 1] = np.round(1 - xy_array_copy[:, 1], 8)
        coordinates = {
            "top": xy_array[0],
            "bottom": xy_array[1],
            "left": xy_array[2],
            "right": xy_array[3],
        }
        for i, xy in enumerate(xy_array[4:]):
            coordinates[f"dart_{i}"] = xy
        return coordinates

    def update_plot(self, xy_array, scores):
        """Update the plot with new coordinates."""
        self.ax[0].clear()
        self.ax[1].clear()

        print(xy_array)
        coordinates = self._map_to_coordinates(xy_array)
        print(coordinates)

        src_points = np.array(
            [
                coordinates["top"],
                coordinates["bottom"],
                coordinates["left"],
                coordinates["right"],
            ]
        )
        dst_points = np.array(
            [
                self.perfect_circle["top"],
                self.perfect_circle["bottom"],
                self.perfect_circle["left"],
                self.perfect_circle["right"],
            ]
        )

        optimized_params = self._compute_transformation_params(src_points, dst_points)
        transformed_arrows = self.transform_all_points(coordinates, optimized_params)

        # Original dartboard plot
        for name, coord in coordinates.items():
            self.ax[0].scatter(coord[0], coord[1], color="red")
            self.ax[0].text(coord[0], coord[1], name, fontsize=9, ha="right")
        self.ax[0].scatter(*zip(*src_points), color="green", label="Calibration Points")
        self.ax[0].set_title("Original Dartboard")
        self.ax[0].set_aspect("equal")
        self.ax[0].invert_yaxis()

        # Transformed dartboard plot
        for name, coord in transformed_arrows.items():
            coord[1] = -coord[1]
            self.ax[1].scatter(coord[0], coord[1], color="blue")
            self.ax[1].text(
                coord[0], coord[1], name, fontsize=9, ha="right", color="blue"
            )
        self.ax[1].scatter(*zip(*dst_points), color="green", label="Calibration Points")
        self.ax[1].add_artist(
            plt.Circle((0, 0), 1, color="blue", fill=False, linestyle="--")
        )
        self.ax[1].set_title("Transformed to Perfect Circle")
        self.ax[1].set_aspect("equal")
        # Add scores to the legend

        img = mpimg.imread(self.bg_img)

        rotated_img = ndimage.rotate(img, -9)
        extent = [-1.4, 1.4, -1.4, 1.4]
        if self.img_displayed:
            self.img_displayed.remove()
        self.img_displayed = self.ax[1].imshow(
            rotated_img, extent=extent, aspect="auto", alpha=1
        )

        score_text = "\n".join(
            [f"Dart {i+1}: {score}" for i, score in enumerate(scores)]
        )
        self.ax[1].legend([score_text], loc="upper right")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
