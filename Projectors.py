import numpy as np

class ScreenProjector():
    """
    Assume the other person is directly on the screen.
    
    Assume full screen.
    """
    def __init__(self, width, height, screen_width, screen_height, intrinsic: np.ndarray) -> None:
        """
        width and height are in pixels. 
        window width and height in meters.
        fov in radian.
        """
        
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.screen_width = screen_width
        self.screen_height = screen_height
        
    def recover_camera_coordinates(self, pixel_coords, distance, log=False):
        """
        pixel_coords: 2D 像素坐标系 (x, y) [shape: (N, 2)]
        
        camera_matrix: camera intrinsic matrix [shape: (3, 3)]
        
        distance: 距离
        """

        # 补全像素坐标
        pixel_coords = np.array(pixel_coords)
        if pixel_coords.ndim == 1:
            pixel_coords = np.expand_dims(pixel_coords, axis=0)
            
        homogeneous_coords = np.concatenate((pixel_coords, np.ones((pixel_coords.shape[0], 1))), axis=1)  # shape: (N, 3)

        # 算回去
        normalized_coords = np.linalg.inv(self.intrinsic) @ homogeneous_coords.T  # shape: (3, N)
        scaled_coords = normalized_coords * distance

        if log:
            print("____________Recover Cam Coords______________")
            print(f"Pixel coords: {pixel_coords}")
            print(f"Inverse intrinsics: {np.linalg.inv(self.intrinsic)}")
            print(f"Recovered coords: {scaled_coords.T}")
        
        return scaled_coords.T  # transpose to shape: (N, 3)
    
    def pixel_coords_to_cam_coords(self, pixel_coords):
        x = - pixel_coords[0] / self.width * self.screen_width + 0.5 * self.screen_width
        y = pixel_coords[1] / self.height * self.screen_height
        return np.array((x, y, 0)) 
    
    def cam_to_head(self, cam_coor):
        return np.array((-cam_coor[0], -cam_coor[2], -cam_coor[1]))
    
    def calculate(self, current_real_coords, remote_pixel_coords, log = True):
        """
        All the coordinates involved here are in real-world coordinates,
        though it doesn't matter in this scenerio.
        
        The head coordinate:
        
        x points from left to right
        
        z points up
        """
        virtual_remote_position = self.pixel_coords_to_cam_coords(remote_pixel_coords)
        gaze_vector = (virtual_remote_position - current_real_coords)[0]
        gaze_vector_head = self.cam_to_head(gaze_vector)
        direction = gaze_vector_head / np.linalg.norm(gaze_vector_head)
        
        if log:
            print("___________Coordinate Calc Details_____________")
            print(f"Current viewpoint cam coord: {current_real_coords}")
            print(f"Virtual remote vector in cam coord: {virtual_remote_position}")        
            print(f"Gaze vector in cam coord: {gaze_vector}")
            print(f"Gaze vector in head coord: {gaze_vector_head}")        
        
        yaw = -np.arctan(-direction[0] / direction[1])
        pitch = np.arctan(-direction[2] / direction[1])
        #pitch = np.arcsin(-direction[2])
        #Assume roll = 0.0
        #print(yaw)
        
        return {
            'yaw': yaw / 3.14 * 180,
            'pitch': pitch / 3.14 * 180,
            'roll': 0.0
        }
        
        
class StraightProjector():
    """
    Assume the other person is a smaller version of themselve except that they be proportionally reduced (denote that by k).
    This k depends on the distance between the user and the screen (both sides)
    
    Assume full screen.
    """
    def __init__(self, width, height, screen_width, screen_height, intrinsic: np.ndarray) -> None:
        """
        width and height are in pixels. 
        window width and height in meters.
        """
        
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        
    def recover_camera_coordinates(self, pixel_coords, distance):
        """
        pixel_coords: 2D 像素坐标系 (x, y) [shape: (N, 2)]
        
        camera_matrix: camera intrinsic matrix [shape: (3, 3)]
        
        distance: 距离
        """
        pixel_coords = np.array(pixel_coords)
        if pixel_coords.ndim == 1:
            pixel_coords = np.expand_dims(pixel_coords, axis=0)
        # 补全像素坐标
        homogeneous_coords = np.concatenate((pixel_coords, np.ones((pixel_coords.shape[0], 1))), axis=1)  # shape: (N, 3)

        # 算回去
        normalized_coords = np.linalg.inv(self.intrinsic) @ homogeneous_coords.T  # shape: (3, N)
        scaled_coords = normalized_coords * distance

        return scaled_coords.T  # transpose to shape: (N, 3)
    
    def to_virtual_coords(self, real_coords, k):
        v_coords = real_coords[0] * k
        return np.array([v_coords[0], -v_coords[1], -v_coords[2]])
    
    def cam_to_head(self, cam_coor):
        return np.array((-cam_coor[0], -cam_coor[2], -cam_coor[1]))
    
    
    def calculate(self, current_real_coords, remote_real_coords, distance, log=True):
        """
        All the coordinates involved here are in real-world coordinates,
        though it doesn't matter in this scenerio.
        
        distance needs to be passed in.
        k = (self.intrinsic[0, 0] / distance) * self.screen_width / self.width
        """
        k = (self.intrinsic[0, 0] / distance) * self.screen_width / self.width
        virtual_remote_position = self.to_virtual_coords(remote_real_coords, k)
        gaze_vector = virtual_remote_position - current_real_coords
        gaze_vector_head = self.cam_to_head(gaze_vector[0])
        direction = gaze_vector_head / np.linalg.norm(gaze_vector_head)
        
        if log:
            print("___________Coordinate Calc Details_____________")
            print(f"k: {k}")
            print(f"Current viewpoint cam coord: {current_real_coords}")
            print(f"Virtual remote vector in cam coord: {virtual_remote_position}")        
            print(f"Gaze vector in cam coord: {gaze_vector}")
            print(f"Gaze vector in head coord: {gaze_vector_head}")   

        yaw = -np.arctan(-direction[0] / direction[1])
        pitch = np.arctan(-direction[2] / direction[1])
        #pitch = np.arcsin(-direction[2])
        #Assume roll = 0.0
        #print(yaw)
        
        return {
            'yaw': yaw / 3.14 * 180,
            'pitch': pitch / 3.14 * 180,
            'roll': 0.0
        }
        
        
class RefractionProjector():
    """
    Assume the other person is behind the screen with a stretch coef k. This time we look at screen as if it was a pool of water with refraction index k.
    Still this k depends on distance(both sides)
       
    Assume full screen.
    """
    def __init__(self, width, height, screen_width, screen_height, intrinsic: np.ndarray) -> None:
        """
        width and height are in pixels. 
        window width and height in meters.
        """
        
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        
    def recover_camera_coordinates(self, pixel_coords, distance):
        """
        pixel_coords: 2D 像素坐标系 (x, y) [shape: (N, 2)]
        
        camera_matrix: camera intrinsic matrix [shape: (3, 3)]
        
        distance: 距离
        """
        pixel_coords = np.array(pixel_coords)
        if pixel_coords.ndim == 1:
            pixel_coords = np.expand_dims(pixel_coords, axis=0)
        # 补全像素坐标
        homogeneous_coords = np.concatenate((pixel_coords, np.ones((pixel_coords.shape[0], 1))), axis=1)  # shape: (N, 3)

        # 算回去
        normalized_coords = np.linalg.inv(self.intrinsic) @ homogeneous_coords.T  # shape: (3, N)
        scaled_coords = normalized_coords * distance

        return scaled_coords.T  # transpose to shape: (N, 3)
    
    def to_virtual_coords(self, real_coords, k):
        v_coords = real_coords[0] * k
        return np.array([v_coords[0], -v_coords[1], -v_coords[2]])
    
    def cam_to_head(self, cam_coor):
        return np.array((-cam_coor[0], -cam_coor[2], -cam_coor[1]))
    
    def calculate(self, current_real_coords, remote_real_coords, log=True):
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
        virtual_remote_coords = self.to_virtual_coords(remote_real_coords, 1)
        gaze_vector = virtual_remote_coords - current_real_coords
        gaze_vector_head = self.cam_to_head(gaze_vector[0])
        direction = gaze_vector_head / np.linalg.norm(gaze_vector_head)

        if log:
            print("___________Coordinate Calc Details_____________")
            print(f"Current viewpoint cam coord: {current_real_coords}")
            print(f"Virtual remote vector in cam coord: {virtual_remote_coords}")        
            print(f"Gaze vector in cam coord: {gaze_vector}")
            print(f"Gaze vector in head coord: {gaze_vector_head}")   
        
        yaw = -np.arctan(-direction[0] / direction[1])
        pitch = np.arctan(-direction[2] / direction[1])
        #pitch = np.arcsin(-direction[2])
        #Assume roll = 0.0
        #print(yaw)
        
        return {
            'yaw': yaw / 3.14 * 180,
            'pitch': pitch / 3.14 * 180,
            'roll': 0.0
        }