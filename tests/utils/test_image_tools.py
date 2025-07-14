from vri.utils.image_tools import convert_images_to_video_b64
import numpy as np

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

if __name__ == "__main__":
    # 运行测试函数
    test_images_to_base64_video()