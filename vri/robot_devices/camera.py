import cv2
from types import SimpleNamespace
import pyudev
import math
from typing import Dict, List, Optional, List
from datetime import datetime, timezone
import time
import numpy as np
import threading
import platform
from pathlib import Path
import websockets.sync.client
from typing import Dict, Optional, Tuple
from websocket import create_connection

class OpenCVCamera:
    """
    The OpenCVCamera class allows to efficiently record images from cameras. It relies on opencv2 to communicate
    with the cameras. Most cameras are compatible. For more info, see the [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

    An OpenCVCamera instance requires a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera
    like a webcam of a laptop, the camera index is expected to be 0, but it might also be very different, and the camera index
    might change if you reboot your computer or re-plug your camera. This behavior depends on your operation system.

    When an OpenCVCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of usage:
    ```python
    camera = OpenCVCamera(camera_index=0)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    camera = OpenCVCamera(0, fps=30, width=1280, height=720)
    camera = connect()  # applies the settings, might error out if these settings are not compatible with the camera

    camera = OpenCVCamera(0, fps=90, width=640, height=480)
    camera = connect()

    camera = OpenCVCamera(0, fps=90, width=640, height=480, color_mode="bgr")
    camera = connect()
    ```
    Attributes:
        camera_index (int): Index of the camera device.
        fps (Optional[int]): Frames per second (FPS) to set for the camera.
        width (Optional[int]): Width of the captured frames.
        height (Optional[int]): Height of the captured frames.
        color_mode (str): Color mode of the captured frames ('rgb' or 'bgr').
        camera (cv2.VideoCapture): OpenCV video capture object.
        is_connected (bool): Flag indicating if the camera is connected.
        thread (Optional[threading.Thread]): Thread for asynchronous reading.
        stop_event (Optional[threading.Event]): Event to stop the reading thread.
        color_image (Optional[np.ndarray]): Latest color image captured.
        logs (Dict[str, Any]): Logs for performance and timestamp information.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """
        Initializes the OpenCVCamera object with provided camera configuration.

        Args:
            config (dict): A dictionary containing camera configuration parameters.
        """
        config = SimpleNamespace(**config)
        self.camera_index = config.camera_index
        self.camera_usb_hardware_index = config.camera_usb_hardware_index
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        self.video_cameras = {}  # 存储打开后的摄像头对象
        self.context = pyudev.Context()
        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

    # 通过usb硬件配置接口连接USB摄像头,链接成功返回摄像头句柄
    def find_rgb_video_device_by_path(self, usb_path: str):
        for device in self.context.list_devices(subsystem="video4linux"):
            id_path = device.get("ID_PATH")
            dev_node = device.device_node
            if id_path and usb_path in id_path:
                print(f" 尝试打开 {dev_node} (ID_PATH: {id_path})")
                cap = cv2.VideoCapture(dev_node)
                if cap.isOpened():
                    print(f" 成功打开 {dev_node} 对应 USB 接口 {usb_path}")
                    return cap  # 一旦成功就退出！
                else:
                    print(f" 无法打开 {dev_node}(USB: {usb_path})")
        return None

    def connect(self) -> None:
        """
        连接摄像头，并根据配置设置分辨率、帧率、颜色模式等参数。

        如果连接失败，会抛出异常（ValueError 或 OSError），提示用户摄像头是否存在或参数是否设置失败。
        """

        # 如果已经连接了，就不允许重复连接
        if self.is_connected:
            raise ValueError(f"OpenCVCamera({self.camera_index}) 已经连接过了。")
        """
        #第一步：尝试用给定的 camera_index 检查摄像头是否可用
        if platform.system() == "Linux":
            # Linux 平台下，摄像头通常是 /dev/videoX 格式的路径
            tmp_camera = cv2.VideoCapture(f"/dev/video{self.camera_index}")
        else:
            # Windows / Mac 平台直接用索引号
            tmp_camera = cv2.VideoCapture(self.camera_index)
        """

        # 检查摄像头是否打开成功
        # is_camera_open = tmp_camera.isOpened()

        # 马上释放临时对象，避免占用资源
        # del tmp_camera

        # 如果摄像头无法打开，尝试提示更清晰的错误信息
        """
        if not is_camera_open:
            # 查看当前系统中可用的摄像头编号
            available_cam_ids = self.find_camera_indices()
            if self.camera_index not in available_cam_ids:
                # 提示使用者传了错误的编号
                raise ValueError(
                    f"`camera_index` 应该是可用的摄像头编号之一 {available_cam_ids}，但你传的是 {self.camera_index}。\n"
                    "请检查摄像头是否插好，或尝试运行 `python lerobot/common/robot_devices/cameras/opencv.py` 来检测摄像头。"
                )

            # 如果编号是正确的但还是打不开，就抛出连接异常
            raise OSError(f"无法访问 OpenCVCamera({self.camera_index})。")
        """

        # 第二步：正式建立连接（刚才只是试探性验证）
        """
        if platform.system() == "Linux":
            self.camera = cv2.VideoCapture(f"/dev/video{self.camera_index}")
        else:
            self.camera = cv2.VideoCapture(self.camera_index)
        """
        # 使用usb硬件接口序号链接摄像头
        self.camera = self.find_rgb_video_device_by_path(self.camera_usb_hardware_index)
        # 第三步：设置摄像头参数（如果有提供）
        if self.fps is not None:
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # 读取实际被设置成功的参数值
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 第四步：校验参数是否真的设置成功（OpenCV 有时会“假装”设置成功）
        if self.fps is not None and not math.isclose(
            self.fps, actual_fps, rel_tol=1e-3
        ):
            raise OSError(
                f"设置帧率失败：期望 {self.fps}，实际是 {actual_fps}（摄像头 {self.camera_index}）"
            )
        if self.width is not None and self.width != actual_width:
            raise OSError(
                f"设置宽度失败：期望 {self.width}，实际是 {actual_width}（摄像头 {self.camera_index}）"
            )
        if self.height is not None and self.height != actual_height:
            raise OSError(
                f"设置高度失败：期望 {self.height}，实际是 {actual_height}（摄像头 {self.camera_index}）"
            )

        # 第五步：设置视频编码格式（MJPG 可大幅提升帧率 & 减少 CPU 压力）
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # if self.camera_usb_hardware_index == "usb-0:2.2":
        #     self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        #     print("设置 MJPG 编码")
        # else:
        #     self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

        # 更新成员变量记录实际成功的参数值
        self.fps = actual_fps
        self.width = actual_width
        self.height = actual_height

        # 标记状态为“已连接”
        self.is_connected = True

    def read(self, temporary_color_mode: Optional[str] = None) -> np.ndarray:
        """Read a frame from the camera and return the frame in the format (height, width, channels).

        Args:
            temporary_color_mode (Optional[str]): Temporary color mode for the frame ('rgb' or 'bgr').

        Returns:
            np.ndarray: The captured color image frame.

        Raises:
            ValueError: If the color mode is invalid.
            OSError: If the camera cannot capture an image or if the image size does not match the expected size.
        """
        if not self.is_connected:
            raise ValueError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        ret, color_image = self.camera.read()
        if not ret:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")

        requested_color_mode = (
            self.color_mode if temporary_color_mode is None else temporary_color_mode
        )

        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # OpenCV uses BGR format as default (blue, green, red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color_mode == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        if h != self.height or w != self.width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = datetime.now(timezone.utc)

        return color_image

    def read_loop(self) -> None:
        """Continuously capture frames in a separate thread."""
        while self.stop_event is None or not self.stop_event.is_set():
            self.color_image = self.read()

    def async_read(self) -> np.ndarray:
        """Asynchronously capture a frame in a separate thread and return it when available.

        Returns:
            np.ndarray: The captured color image frame.

        Raises:
            ValueError: If the camera is not connected.
            Exception: If the thread for asynchronous reading fails to start.
        """
        if not self.is_connected:
            raise ValueError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()
            ## zq
            time.sleep(1)  # give some time for the thread to start

        num_tries = 0
        while self.color_image is None:
            num_tries += 1
            time.sleep(1 / self.fps)
            if num_tries > self.fps and (
                self.thread.ident is None or not self.thread.is_alive()
            ):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )

        return self.color_image

    def disconnect(self) -> np.ndarray:
        """Disconnect the camera, stop the thread, and release the resources."""
        if not self.is_connected:
            raise ValueError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.release()
        self.camera = None

        self.is_connected = False

    def __del__(self) -> None:
        """Ensure proper cleanup of resources when the object is destroyed."""
        if getattr(self, "is_connected", False):
            self.disconnect()

    def find_camera_indices(
        raise_when_empty=False, max_index_search_range=60
    ) -> List[int]:
        """
        Finds available camera indices by scanning the system for connected cameras.

        On Linux, it scans the '/dev/video*' ports to find the camera indices.
        On macOS and Windows, it tries camera indices from 0 to `max_index_search_range`.

        Args:
            raise_when_empty (bool): Whether to raise an exception if no camera is found. Default is False.
            max_index_search_range (int): The maximum index range to search for cameras on non-Linux platforms (default is 60).

        Returns:
            list[int]: A list of available camera indices.

        Raises:
            OSError: If no cameras are found and `raise_when_empty` is set to True.

        Notes:
            - On Linux, the function looks for camera devices in the '/dev' directory (e.g., '/dev/video0', '/dev/video1', etc.).
            - On non-Linux platforms, it checks camera indices from 0 up to `max_index_search_range`.
            - Cameras that are accessible through OpenCV are identified by attempting to open each camera index.
        """
        if platform.system() == "Linux":
            # Linux uses camera ports
            print(
                "Linux detected. Finding available camera indices through scanning '/dev/video*' ports"
            )
            possible_camera_ids = []
            for port in Path("/dev").glob("video*"):
                camera_idx = int(str(port).replace("/dev/video", ""))
                possible_camera_ids.append(camera_idx)
        else:
            print(
                "Mac or Windows detected. Finding available camera indices through "
                f"scanning all indices from 0 to {60}"
            )
            possible_camera_ids = range(max_index_search_range)

        camera_ids = []
        for camera_idx in possible_camera_ids:
            camera = cv2.VideoCapture(camera_idx)
            is_open = camera.isOpened()
            camera.release()

            if is_open:
                print(f"Camera found at index {camera_idx}")
                camera_ids.append(camera_idx)

        if raise_when_empty and len(camera_ids) == 0:
            raise OSError(
                "Not a single camera was detected. Try re-plugging, or re-installing `opencv2`, "
                "or your camera driver, or make sure your camera is compatible with opencv2."
            )

        return camera_ids
    
class WbSocketCamera:
    def __init__(self, config: dict, **kwargs) -> None:
        """
        Initializes the WbSocketCamera object with provided camera configuration.

        Args:
            config (dict): A dictionary containing camera configuration parameters.
        """
        config = SimpleNamespace(**config)
        self.camera_index = config.camera_index
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        host = "192.168.0.210"
        port = "10002"
        self._uri = f"ws://{host}:{port}"
        self.is_connected = False
        self.logs = {}
        self._ws = None
        # self._ws, self._server_metadata = self._wait_for_server()
    
    def connect(self) -> None:
        """
        连接摄像头，并根据配置设置分辨率、帧率、颜色模式等参数。

        如果连接失败，会抛出异常（ValueError 或 OSError），提示用户摄像头是否存在或参数是否设置失败。
        """

        # 如果已经连接了，就不允许重复连接
        if self.is_connected:
            raise ValueError(f"WbSocketCamera({self.camera_index}) 已经连接过了。")
        
        self._ws = create_connection(self._uri)
        self.is_connected = True
    
    def async_read(self) -> np.ndarray:
        """Asynchronously capture a frame in a separate thread and return it when available.

        Returns:
            np.ndarray: The captured color image frame.
        """
        start_time = time.perf_counter()
        self._ws.send("get_frame")
        response = self._ws.recv()
        if isinstance(response, str):
            print("Received string response from server:", response)
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        img_array = np.frombuffer(response, np.uint8).reshape((self.height, self.width, 3))
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        return img_array
    
    def disconnect(self) -> None:
        """Disconnect the camera, stop the thread, and release the resources."""
        if not self.is_connected:
            raise ValueError(
                f"WbSocketCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )
        self._ws.close()
        self.is_connected = False

if __name__ == "__main__":
    cameras = {
        "cam_high":{
            "camera_index": 4,
            "fps": 60,
            "width": 640,
            "height": 480,
            "color_mode": "rgb",
            "camera_usb_hardware_index": "usb-0:2.4",
            "type": "opencv"
       },
        "CAM_LEFT_WRIST":{
            "camera_index": 2,
            "fps": 25,
            "width": 640,
            "height": 480,
            "color_mode": "rgb",
            "camera_usb_hardware_index": "usb-0:4.2.4",
            "type": "opencv"
            },
        
        "cam_right_wrist":{
            "camera_index": 6,
            "fps": 25,
            "width": 640,
            "height": 480,
            "color_mode": "rgb",
            "camera_usb_hardware_index": "usb-0:4.4.3",
            "type": "opencv"
        }
    }
    # 连接所有摄像头
    camera_objects = {}
    for camera_name, camera_config in cameras.items():
        try:
            camera = OpenCVCamera(camera_config)
            camera.connect()
            camera_objects[camera_name] = camera
            print(f"成功连接摄像头: {camera_name}")
        except Exception as e:
            print(f"连接摄像头 {camera_name} 失败: {e}")
    

    # 显示摄像头画面
    try:
        while True:
            frames = {}
            
            # 读取所有摄像头的画面
            for camera_name, camera in camera_objects.items():
                try:
                    # 读取图像
                    frame = camera.read()
                    
                    # 由于OpenCV显示需要BGR格式，如果配置是RGB需要转换
                    if camera.color_mode == "rgb":
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # 添加摄像头名称标签
                    cv2.putText(frame, camera_name, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    frames[camera_name] = frame
                    
                except Exception as e:
                    print(f"读取摄像头 {camera_name} 失败: {e}")
                    # 创建一个黑色画面作为占位符
                    frames[camera_name] = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frames[camera_name], f"{camera_name}: Error", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 根据摄像头数量创建不同的布局
            num_cameras = len(frames)
            
            if num_cameras == 1:
                # 只有一个摄像头，直接显示
                combined_frame = list(frames.values())[0]
                
            elif num_cameras == 2:
                # 两个摄像头，水平排列
                frame_list = list(frames.values())
                combined_frame = np.hstack(frame_list)
                
            elif num_cameras == 3:
                # 三个摄像头，2x2网格（第四个位置留空）
                frame_list = list(frames.values())
                # 前两个水平排列
                top_row = np.hstack(frame_list[:2])
                # 第三个放在第二行中间
                bottom_row = np.hstack([frame_list[2], np.zeros_like(frame_list[2])])
                combined_frame = np.vstack([top_row, bottom_row])
                
            elif num_cameras == 4:
                # 四个摄像头，2x2网格
                frame_list = list(frames.values())
                top_row = np.hstack(frame_list[:2])
                bottom_row = np.hstack(frame_list[2:4])
                combined_frame = np.vstack([top_row, bottom_row])
                
            else:
                # 多于4个摄像头，创建网格布局
                frame_list = list(frames.values())
                cols = 2  # 每行2列
                rows = (num_cameras + cols - 1) // cols  # 计算需要的行数
                
                # 创建空白画布
                combined_height = rows * 480
                combined_width = cols * 640
                combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
                
                # 将每个画面放置到网格中
                for i, frame in enumerate(frame_list):
                    row = i // cols
                    col = i % cols
                    y_start = row * 480
                    y_end = y_start + 480
                    x_start = col * 640
                    x_end = x_start + 640
                    combined_frame[y_start:y_end, x_start:x_end] = frame
            
            # 显示合并后的画面
            cv2.imshow('Multiple Cameras', combined_frame)
            
            # 检测按键，按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # 添加短暂延迟以减少CPU使用率
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        # 断开所有摄像头连接
        for camera_name, camera in camera_objects.items():
            try:
                camera.disconnect()
                print(f"已断开摄像头: {camera_name}")
            except Exception as e:
                print(f"断开摄像头 {camera_name} 失败: {e}")

    # camera_config = {
    #         "camera_index": 2,
    #         "fps": 60,
    #         "width": 640,
    #         "height": 480,
    #         "color_mode": "rgb",
    #         "camera_usb_hardware_index": "usb-0:2.1",
    #         "type": "wb_socket"
    # }
    # camera = WbSocketCamera(camera_config)
    # camera.connect()
    # import time
    # start = time.time()
    # data = camera.async_read()
    # end = time.time()
    # print("exec time:", end - start)
    # print("data type:", type(data), "data shape:", data.shape)
    # print("data:", data)
    # cv2.imwrite('/home/qluan/users/quebinbin/workspace/projects/vla_robots_interface/opencv_image.png', data)
    # camera.disconnect()
