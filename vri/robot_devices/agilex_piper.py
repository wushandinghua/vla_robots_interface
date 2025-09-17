from types import SimpleNamespace
import copy
import time
from typing import Dict

from vri.utils.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST
from vri.robot_devices.camera import OpenCVCamera
from piper_sdk import C_PiperInterface_V2 as PiperArm



_PIPER_ROBOT_CONFIG = {
    "follower_number": 2,
    "follower_can": ["can_left", "can_right"],
    "start_arm_joint_position": [
        0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0
    ],
    "cameras": {
        CAM_HIGH:{
            "camera_index": 4,
            "fps": 60,
            "width": 640,
            "height": 480,
            "color_mode": "rgb",
            "camera_usb_hardware_index": "usb-0:1.4"
       },
        CAM_LEFT_WRIST:{
            "camera_index": 2,
            "fps": 25,
            "width": 640,
            "height": 480,
            "color_mode": "rgb",
            "camera_usb_hardware_index": "usb-0:4.2.4"
            },
        
        CAM_RIGHT_WRIST:{
            "camera_index": 6,
            "fps": 25,
            "width": 640,
            "height": 480,
            "color_mode": "rgb",
            "camera_usb_hardware_index": "usb-0:4.4.3"
            }
    }
}


class PiperPlay:
    """
    A class to manage the Piper robot and cameras for data collection and robotic control.
    This class handles the initialization, mode switching, and data capture from multiple cameras
    and robotic arms (leader and follower robots).
    """

    def __init__(self, reset_type, reset_position) -> None:
        """
        Initializes the PiperPlay object by connecting to cameras and robots.

        Args:
            reset_type, 
            reset_position: 0: no reset, 1: reset when robot connect, 2: reset when robot disconnect, 3: reset when robot connect and disconnect
        """
        self.config = SimpleNamespace(**copy.deepcopy(_PIPER_ROBOT_CONFIG))
        self.cameras = self.config.cameras
        print("Piper config:", self.config)
        
        self._reset_type = reset_type
        self._reset_position = reset_position or self.config.start_arm_joint_position
        print("reset_type:", self._reset_type, "reset_position:", self._reset_position)
        # Initialize cameras
        for name in self.cameras:
            self.cameras[name] = OpenCVCamera(self.cameras[name])
        
        self.logs = {}
        self.is_connected = False
        follower_robot = []
        """
        初始化 follower 机器人
        """
        args = self.config  
        for i in range(args.follower_number):
            follower_robot.append(
                PiperArm(can_name=args.follower_can[i])
            )
            time.sleep(0.1)
            print(f"follower robot {i} 初始化成功")

        self.follower_robot = follower_robot
        time.sleep(0.3)

        try:
            self.connect()
        except Exception as e:
            print("Failed to initialize Piper, err:", e)

    def send_action(self, target_joint_positions, is_gripper_work=True):
        assert len(target_joint_positions) // 7 == len(self.follower_robot)
        if not isinstance(target_joint_positions, list):
            target_joint_positions = list(target_joint_positions)
        # print(target_joint_positions)
        # 这里假设非阻塞执行，视为并发控制
        # print("send_action:", target_joint_positions)
        for i in range(len(self.follower_robot)):
            joint_0 = int(round(target_joint_positions[i * 7 + 0]))
            joint_1 = int(round(target_joint_positions[i * 7 + 1]))
            joint_2 = int(round(target_joint_positions[i * 7 + 2]))
            joint_3 = int(round(target_joint_positions[i * 7 + 3]))
            joint_4 = int(round(target_joint_positions[i * 7 + 4]))
            joint_5 = int(round(target_joint_positions[i * 7 + 5]))
            joint_6 = int(round(target_joint_positions[i * 7 + 6]))
            self.follower_robot[i].MotionCtrl_2(0x01, 0x01, 100, 0x00)
            self.follower_robot[i].JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            if is_gripper_work:
                self.follower_robot[i].GripperCtrl(joint_6, 1000, 0x01, 0)

    def connect(self):
        if self.is_connected:
            print("Piper is already connected. Do not run `robot.connect()` twice.'")
            raise ConnectionError()
        
         # Connect the cameras
        for name in self.cameras:
            print(f"Connecting to camera: {name}")
            self.cameras[name].connect()
            ## zq
            # if name == CAM_HIGH:
            #     time.sleep(0.5)  # Wait a bit for the camera to initialize

        """
        设置控制模式和初始关节位置
        """

        for i, robot in enumerate(self.follower_robot):
            robot.ConnectPort()
            while( not robot.EnablePiper()):
                time.sleep(0.01)
            
        if self._reset_type & 1 > 0:
            self.send_action(self._reset_position)
            # self.send_action(self._reset_position, is_gripper_work=False)
        
        
        self.is_connected = True
    
    def disconnect(self) -> None:
        """
        Disconnects the cameras and cleans up resources.
        """
        if not self.is_connected:
            print("Piper is not connected. You need to run `robot.connect()` before disconnecting.'")
            raise ConnectionError()
        
        try:
            for name in self.cameras:
                self.cameras[name].disconnect()
            
            if self._reset_type & 2 > 0:
                self.send_action([0] * 14, is_gripper_work=False)
            
            for i in range(self.config.follower_number):
                self.follower_robot[i].DisconnectPort()
        except Exception as e:
            print("Failed to disconnect Piper, err:", e)
        self.is_connected = False
        print("Robot exited")


    # 获取机器人当前状态（低维度数据）
    def get_low_dim_data(self) -> Dict[str, any]:
        """
        收集 follower 机器人的低维状态数据。
        """
        args = self.config
        follower_robot = self.follower_robot

        def get_joint_positions(joint_state):
            joints = [joint_state.joint_1, joint_state.joint_2, joint_state.joint_3,
              joint_state.joint_4, joint_state.joint_5, joint_state.joint_6]
            return joints

        def get_gripper_position(gripper_state):
            return gripper_state.grippers_angle
        
        data = {}
        state = []
        # print("获取 Follower 机器人状态...")
        for i in range(args.follower_number):
            join_msg = follower_robot[i].GetArmJointMsgs()
            state.extend(get_joint_positions(join_msg.joint_state))
            gripper_msg = follower_robot[i].GetArmGripperMsgs()
            state.append(get_gripper_position(gripper_msg.gripper_state))
        data["observation.state"] = state
        return data

    # 从相机读取图像帧（图像观察数据） ， 从机器人读取当前状态（低维度状态数据）
    def capture_observation(self) -> Dict[str, any]:
        """
        Captures observations including both images from cameras and low-dimensional data.

        Returns:
            A dictionary containing the time stamps, low-dimensional data, and camera images.
        """
        obs_act_dict = {}
        images = {}

        for name in self.cameras:
            before_camread_t = time.perf_counter()
            img = self.cameras[name].async_read()  # 异步读取图像
            # print(f"[DEBUG] capture_observation: {name} type: {type(img)} shape: {getattr(img, 'shape', None)}") #这边打印的图像类型是没有问题的
            images[name] = img  # 将图像置入列表中

            #obs_act_dict[f"/time/{name}"] = time.time()  # 添加时间戳
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        low_dim_data = self.get_low_dim_data()

        # Populate output dictionaries
        obs_act_dict.update(low_dim_data)  # 同时记录机械臂低维度的数据
        for name in self.cameras:
            obs_act_dict[f"observation.images.{name}"] = images[
                name
            ]  # 将读取到的图像转存储到obs列表中（要在这里检查图像类型吗）
        return obs_act_dict
                
    
    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

if __name__ == "__main__":
    robot = PiperPlay(reset_type=1, reset_position=None)
    #robot.connect()
    robot.send_action([0.0] * 6 + [80000] + [0.0] * 6 + [80000])
    #robot.send_action([-425, -196, -12000, 53000, 9000, -40000, 0, -425, -196, -12000, 53000, 9000, -40000, 0])
    #robot.send_action([-425, -196, -12000, 53000, 9000, -40000, 0])
    time.sleep(3) 
    #print("low data:", robot.get_low_dim_data())
    print("obs1:", robot.capture_observation())
    robot.send_action([0.0] * 6 + [300] + [0.0] * 6 + [60000])
    #robot.send_action([0.0] * 6 + [40000]) 
    time.sleep(2)
    #robot.send_action([0.0] * 6 + [0] + [0.0] * 6 + [0])
    #robot.send_action([0.0] * 6 + [0]) 
    time.sleep(1)
    print("obs2:", robot.capture_observation())
    robot.disconnect()
