from vri.utils.image_tools import convert_images_to_video_b64, concat_images_horizontally
import numpy as np
from PIL import Image

def test_images_to_base64_video():
    """
    Test function to convert a list of images to a Base64 video string.
    """
    # 生成一些随机图像数据
    # 这里假设每个图像是128x128的RGB图像
    images = [np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(10)] 
    # 转换为Base64视频
    base64_video = convert_images_to_video_b64(images, fps=5)
    
    print(f"生成Base64视频字符串 (长度: {len(base64_video)} 字符)")
    print("前100字符:", base64_video[:100] + "...")

def test_concat_images_horizontally():
    """
    Test function to concatenate images from three cameras horizontally.
    """
    # 生成一些随机图像数据
    cam_high_images = [np.random.randint(0, 1, (128, 128, 3), dtype=np.uint8) for _ in range(1)]
    cam_left_wrist_images = [np.random.randint(128, 129, (128, 128, 3), dtype=np.uint8) for _ in range(1)]
    cam_right_wrist_images = [np.random.randint(255, 256, (128, 128, 3), dtype=np.uint8) for _ in range(1)]
    
    # 合并图像
    concatenated_images = concat_images_horizontally(cam_high_images, cam_left_wrist_images, cam_right_wrist_images)
    
    print(f"合并后的图像数量: {len(concatenated_images)}")
    print("合并后的第一张图像形状:", concatenated_images[0].shape)
    # left img
    img = Image.fromarray(cam_left_wrist_images[0])
    img.show(title="Left Wrist Camera Image")
    img = Image.fromarray(cam_high_images[0])
    img.show(title="High Camera Image")
    img = Image.fromarray(cam_right_wrist_images[0])
    img.show(title="Right Wrist Camera Image")
    img = Image.fromarray(concatenated_images[0])
    img.show(title="Concat Camera Image")


if __name__ == "__main__":
    # 运行测试函数
    # test_images_to_base64_video()
    test_concat_images_horizontally()