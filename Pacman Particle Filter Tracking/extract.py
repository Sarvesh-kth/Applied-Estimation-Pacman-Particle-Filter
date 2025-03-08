"""
Code to extract all relevant information from the current frame
"""

import cv2
import numpy as np


class Extract:
    """
    Extracts all relevant information from the current frame
    """

    def __init__(self):
        self.object_colors = {
            "pacman": {"lower": (20, 100, 100), "upper": (30, 255, 255)},  # Yellow
            "red_ghost": {"lower": (0, 100, 100), "upper": (10, 255, 255)},  # Red
            "cyan_ghost": {"lower": (80, 100, 100), "upper": (100, 255, 255)},  # Cyan
            "pink_ghost": {"lower": (120, 50, 60), "upper": (150, 255, 255)},  # Pink
            "orange_ghost": {"lower": (0, 100, 100), "upper": (20, 255, 255)},  # Orange
        }
        self.ghost_marker_colors = {
            "pacman": (0, 255, 255),
            "red_ghost": (0, 0, 255),
            "cyan_ghost": (255, 255, 0),
            "pink_ghost": (255, 0, 255),
            "orange_ghost": (0, 165, 255),
        }
        self.movements = {
            1: "up",
            -1: "down",
            -10: "left",
            10: "right",
            11: "upright",
            -11: "downleft",
            9: "upleft",
            -9: "downright",
            0: "nomo",
        }

    def find_center(self, mask: cv2.Mat):
        """
        Find the center of the object in the mask

        Args:
            mask: binary mask of the object

        Returns:
            (cx, cy): center of the object
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None
    
    def find_all_centers(self, mask: cv2.Mat):
        """
        Find all centers of the object in the mask
        
        Args:
            mask: binary mask of the object
            
        Returns:
            centers: list of centers of the object
        """

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
        return centers

    def bgr_to_hsv(self, frame: cv2.Mat):
        """converts BGR image to HSV image"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def extract_locations(self, frame: cv2.Mat, multi=False, remove_spawn_point=True):
        """
        Extracts the locations of all objects in the frame

        Args:
            frame: current frame in BGR
            multi: whether to extract multiple objects of the same type
            remove_spawn_point: whether to remove the spawn point from the frame

        Returns:
            positions: dictionary containing the positions of all objects
        """
        positions = {}
        if remove_spawn_point:
            frame[220:240, 170:190, :] = 0
        hsv = self.bgr_to_hsv(frame)
        for name, color in self.object_colors.items():
            mask = cv2.inRange(hsv, np.array(color["lower"]), np.array(color["upper"]))
            if multi:
                center = self.find_all_centers(mask)
            else:
                center = self.find_center(mask)
            positions[name] = center
        return positions

    def extract_movement(self, positions: dict, prev_positions: dict):
        """
        Extracts the movement of all objects in the frame

        Args:
            frame: current frame in BGR
            prev_positions: dictionary containing the positions of all objects in the previous frame

        Returns:
            movements: dictionary containing the movements of all objects
        """
        movements = {}
        for name in positions:
            if name in prev_positions:
                prev_position = prev_positions[name]
                if prev_position is not None and positions[name] is not None:
                    movements[name] = self.movements[
                        np.sign(positions[name][0] - prev_position[0]) * 10
                        + np.sign(positions[name][1] - prev_position[1])
                    ]
                else:
                    movements[name] = None
            else:
                movements[name] = None
        return movements
    

def valid_pacman_movements(self, frame: cv2.Mat, pacman_center: tuple, offset=20):
        """
        Checks for walls in the immediate vicinity of Pac-Man (up, down, left, right).

        Args:
            frame: Current frame in BGR
            pacman_center: (cx, cy) center of Pac-Man
            offset: How many pixels away from the center to check for a wall

        Returns:
            A dictionary indicating whether pacman can move in each direction.
            e.g. {"up": True, "down": False, "left": True, "right": False}
        """
        # If we have no detected Pac-Man, just return False in all directions
        if pacman_center is None:
            return {"up": False, "down": False, "left": False, "right": False}

        hsv = self.bgr_to_hsv(frame)

        # Define a rough HSV range for the blue walls (you may need to adjust these)
        walls_lower = (100, 100, 100)  
        walls_upper = (130, 255, 255)

        # Create a mask that highlights the walls
        walls_mask = cv2.inRange(hsv, walls_lower, walls_upper)

        cx, cy = pacman_center
        height, width = walls_mask.shape
        
        # Safeguard boundaries
        up_y = max(cy - offset, 0)
        down_y = min(cy + offset, height - 1)
        left_x = max(cx - offset, 0)
        right_x = min(cx + offset, width - 1)

        # Check pixel values in rectangle in each direction

        up_wall_matrix = walls_mask[up_y:cy-10, cx-10:cx+10]
        down_wall_matrix = walls_mask[cy+10:down_y, cx-10:cx+10]
        left_wall_matrix = walls_mask[cy-10:cy+10, left_x:cx-10]
        right_wall_matrix = walls_mask[cy-10:cy+10, cx+10:right_x]

        up_wall = ~np.any(up_wall_matrix)
        down_wall = ~np.any(down_wall_matrix)
        left_wall = ~np.any(left_wall_matrix)
        right_wall = ~np.any(right_wall_matrix)

        return {
            "up": up_wall,
            "down": down_wall,
            "left": left_wall,
            "right": right_wall
        }
