import cv2
import numpy as np

class ScreenProjector():
    """
    Assume the other person is directly on the screen.
    
    Assume full screen.
    """
    def __init__(self, width, height, screen_width, screen_height, intrinsic: np.ndarray, reference, distance) -> None:
        """
        width and height are in pixels. 
        window width and height in meters.
        fov in radian.
        """
        
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.reference = reference
        self.real_world_reference = self.recover_camera_coordinates(reference, intrinsic, distance)
        self.screen_width = screen_width
        self.screen_height = screen_height
        
    def recover_camera_coordinates(self, pixel_coords, camera_matrix, distance):
        """
        pixel_coords: 2D 像素坐标系 (x, y) [shape: (N, 2)]
        
        camera_matrix: camera intrinsic matrix [shape: (3, 3)]
        
        distance: 距离
        """

        # 补全像素坐标
        homogeneous_coords = np.concatenate((pixel_coords, np.ones((pixel_coords.shape[0], 1))), axis=1)  # shape: (N, 3)

        # 算回去
        normalized_coords = np.linalg.inv(camera_matrix) @ homogeneous_coords.T  # shape: (3, N)
        scaled_coords = normalized_coords * distance

        return scaled_coords.T  # transpose to shape: (N, 3)
    
    def pixel_coords_to_screen_coords(self, pixel_coords):
        x = pixel_coords[0] / self.width * self.screen_width
        y = pixel_coords[1] / self.height * self.screen_height
        return np.array((x, y, 0)) 
    
    def cam_to_head(cam_coor):
        return np.array((-cam_coor[0], -cam_coor[2], -cam_coor[1]))
    
    def calculate(self, current_real_coords, remote_pixel_coords):
        """
        All the coordinates involved here are in real-world coordinates,
        though it doesn't matter in this scenerio.
        """
        virtual_remote_position = self.pixel_coords_to_screen_coords(remote_pixel_coords)
        gaze_vector = virtual_remote_position - current_real_coords
        gaze_vector_head = self.cam_to_head(gaze_vector)
        direction = gaze_vector_head / np.linalg.norm(gaze_vector_head)
        
        yaw = -np.arctan2(direction[1], direction[0])
        pitch = np.arcsin(direction[1])
        # Assume roll = 0.0
        
        return {
            'yaw': np.degrees(yaw),
            'pitch': np.degrees(pitch),
            'roll': 0.0
        }
        
        
class StraightProjector():
    """
    Assume the other person is a smaller version of themselve except that they be proportionally reduced (denote that by k).
    This k depends on the distance between the user and the screen (both sides)
    
    Assume full screen.
    """
    def __init__(self, width, height, screen_width, screen_height, intrinsic: np.ndarray, reference, distance) -> None:
        """
        width and height are in pixels. 
        window width and height in meters.
        fov in radian.
        """
        
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.reference = reference
        self.real_world_reference = self.recover_camera_coordinates(reference, intrinsic, distance)
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        
    def recover_camera_coordinates(self, pixel_coords, camera_matrix, distance):
        """
        pixel_coords: 2D 像素坐标系 (x, y) [shape: (N, 2)]
        
        camera_matrix: camera intrinsic matrix [shape: (3, 3)]
        
        distance: 距离
        """

        # 补全像素坐标
        homogeneous_coords = np.concatenate((pixel_coords, np.ones((pixel_coords.shape[0], 1))), axis=1)  # shape: (N, 3)

        # 算回去
        normalized_coords = np.linalg.inv(camera_matrix) @ homogeneous_coords.T  # shape: (3, N)
        scaled_coords = normalized_coords * distance

        return scaled_coords.T  # transpose to shape: (N, 3)
    
    def to_virtual_coords(self, real_coords, k):
        v_coords = real_coords * k
        return np.array([v_coords[0], -v_coords[1], -v_coords[2]])
    
    def cam_to_head(cam_coor):
        return np.array((-cam_coor[0], -cam_coor[2], -cam_coor[1]))
    
    def calculate(self, current_real_coords, remote_real_coords, distance):
        """
        All the coordinates involved here are in real-world coordinates,
        though it doesn't matter in this scenerio.
        
        distance needs to be passed in.
        k = (self.intrinsic[0, 0] / distance) * self.screen_width / self.width
        """
        k = (self.intrinsic[0, 0] / distance) * self.screen_width / self.width
        virtual_remote_position = self.to_virtual_coords(remote_real_coords, k)
        gaze_vector = virtual_remote_position - current_real_coords
        gaze_vector_head = self.cam_to_head(gaze_vector)
        direction = gaze_vector_head / np.linalg.norm(gaze_vector_head)
        
        yaw = -np.arctan2(direction[1], direction[0])
        pitch = np.arcsin(direction[1])
        # Assume roll = 0.0
        
        return {
            'yaw': np.degrees(yaw),
            'pitch': np.degrees(pitch),
            'roll': 0.0
        }
        
        
class RefractionProjector():
    """
    Assume the other person is behind the screen with a stretch coef k. This time we look at screen as if it was a pool of water with refraction index k.
    Still this k depends on distance(both sides)
       
    Assume full screen.
    """
    def __init__(self, width, height, screen_width, screen_height, intrinsic: np.ndarray, reference, distance) -> None:
        """
        width and height are in pixels. 
        window width and height in meters.
        fov in radian.
        """
        
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.reference = reference
        self.real_world_reference = self.recover_camera_coordinates(reference, intrinsic, distance)
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        
    def recover_camera_coordinates(self, pixel_coords, camera_matrix, distance):
        """
        pixel_coords: 2D 像素坐标系 (x, y) [shape: (N, 2)]
        
        camera_matrix: camera intrinsic matrix [shape: (3, 3)]
        
        distance: 距离
        """

        # 补全像素坐标
        homogeneous_coords = np.concatenate((pixel_coords, np.ones((pixel_coords.shape[0], 1))), axis=1)  # shape: (N, 3)

        # 算回去
        normalized_coords = np.linalg.inv(camera_matrix) @ homogeneous_coords.T  # shape: (3, N)
        scaled_coords = normalized_coords * distance

        return scaled_coords.T  # transpose to shape: (N, 3)
    
    def to_virtual_coords(self, real_coords, k):
        v_coords = real_coords * k
        return np.array([v_coords[0], -v_coords[1], -v_coords[2]])
    
    def cam_to_head(cam_coor):
        return np.array((-cam_coor[0], -cam_coor[2], -cam_coor[1]))
    
    def calculate(self, current_real_coords, remote_real_coords, distance):
        """
        All the coordinates involved here are in real-world coordinates,
        though it doesn't matter in this scenerio.
        
        distance needs to be passed in.
        k = (self.intrinsic[0, 0] / distance) * self.screen_width / self.width
        """
        
            
        # Now that we have two points, we calculate the minimum distance between the two.
        # We look for a path of light between the two.
        # sin(theta_{otherside}) / sin(theta_ourside) = k
        # However, after a bit of simple math we can find that 
        # the intersection of the light and the screen is the intersection point 
        # when assuming a 1:1 placement of the opposite space behind the screen 
        # and not taking refraction into account.
        remote_virtual_coords = self.to_virtual_coords(remote_real_coords, 1)
        gaze_vector = remote_virtual_coords - current_real_coords
        gaze_vector_head = self.cam_to_head(gaze_vector)
        direction = gaze_vector_head / np.linalg.norm(gaze_vector_head)
        
        yaw = -np.arctan2(direction[1], direction[0])
        pitch = np.arcsin(direction[1])
        # Assume roll = 0.0
        
        return {
            'yaw': np.degrees(yaw),
            'pitch': np.degrees(pitch),
            'roll': 0.0
        }