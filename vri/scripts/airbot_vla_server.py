from flask import Flask, request, jsonify
from vri.scripts import runtime as _runtime
from vri.subscribers import video_display as _video_display
from vri.environments import airbot_env as _airbot_env 
from vri.agents import policy_agent as _policy_agent
from vri.policies import action_chunk_policy as _action_chunk_policy
from vri.policies import websocket_client_policy as _websocket_client_policy
from vri.utils.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST
from vri.utils import image_tools
import time
import threading
import logging
logging.getLogger().setLevel(logging.INFO)
import numpy as np

"""
conda activate airbot_5_8
pip install flask==3.1.1
pip install imageio==2.37.0
pip install imageio-ffmpeg==0.6.0
export PYTHONPATH=/home/zq/work/fnii/vla_robots_interface:$PYTHONPATH
python vri/scripts/airbot_vla_server.py
"""

app = Flask(__name__)
remote_model_ip = "192.168.3.101"
ws_client_tap = _websocket_client_policy.WebsocketClientPolicy(host=remote_model_ip, port=8001)
ws_client_open_oven = _websocket_client_policy.WebsocketClientPolicy(host=remote_model_ip, port=8002)
ws_client_close_oven = _websocket_client_policy.WebsocketClientPolicy(host=remote_model_ip, port=8003)
ws_client_others = _websocket_client_policy.WebsocketClientPolicy(host=remote_model_ip, port=8000)
ws_client_dict = {
    "tap": ws_client_tap,
    "open_oven": ws_client_open_oven,
    "close_oven": ws_client_close_oven,
    "others": ws_client_others
}
# warmup
fake_observation = {
            CAM_HIGH: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            CAM_LEFT_WRIST: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            CAM_RIGHT_WRIST: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "state": np.random.rand(14),
            "prompt": "do something",
}
for k, ws_client in ws_client_dict.items():
    ws_client.infer(fake_observation)

# cache images of cam high
cam_high_images = []
capture_thread = None
capture_running = False

def capture_cam_high_images():
    global cam_high_images, capture_running
    from vri.robot_devices.airbot import OpenCVCamera, _ROBOT_CONFIG
    cam_high = OpenCVCamera(_ROBOT_CONFIG["cameras"][CAM_HIGH])
    cam_high.connect()
    time.sleep(0.1)  # 等待相机连接稳定
    while capture_running:
        frame = cam_high.read()
        cam_high_images.append(frame)
        time.sleep(1 / _ROBOT_CONFIG["cameras"][CAM_HIGH]["fps"])  # Capture at 30 FPS
    cam_high.disconnect()

@app.route('/vla', methods=['POST'])
def vla():
    data = request.json
    logging.info(f"received request: {data}")
    instruction = data.get('action')
    if 'tap' in instruction:
        ws_client_policy = ws_client_dict['tap']
    elif 'open' in instruction and 'oven' in instruction:
        ws_client_policy = ws_client_dict['open_oven']
    elif 'close' in instruction and 'oven' in instruction:
        ws_client_policy = ws_client_dict['close_oven']
    else:
        ws_client_policy = ws_client_dict['others']
    logging.info(f"instruction: {instruction}, uri: {ws_client_policy._uri}")
    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    runtime = _runtime.Runtime(
        environment=_airbot_env.AirbotEnvironment(reset_position=metadata.get("reset_pose"), instruction=instruction, reset_type=1),
        agent=_policy_agent.PolicyAgent(
            policy=_action_chunk_policy.ActionChunkPolicy(
                policy=ws_client_policy,
                action_chunk_size=10
            )
        ),
        subscriber=_video_display.VideoDisplay([CAM_LEFT_WRIST, CAM_HIGH, CAM_RIGHT_WRIST]),
        max_hz=50,
        num_episodes=1,
        max_episode_steps=15*50
    )

    action_status_type, video_b64 = runtime.run()
    data = {
        "action_status_type": action_status_type,
        "video_b64": video_b64
    }
    return jsonify({"status": "success", "message": "robot action complete", "data": data}), 200

@app.route('/video', methods=['POST'])
def video():
    global cam_high_images, capture_thread, capture_running
    
    data = request.json
    logging.info(f"received request: {data}")
    instruction = data.get('action')
    if 'start' in instruction:
        cam_high_images = []
        capture_running = True
        """read images """
        capture_thread = threading.Thread(target=capture_cam_high_images)
        capture_thread.start()
        logging.info("video capture started")
        return jsonify({"status": "success", "message": "video capture start", "data": {}}), 200
    elif 'stop' in instruction:
        capture_running = False
        if capture_thread is not None:
            capture_thread.join()
        logging.info("video capture finished")
        # Convert images to video
        video_b64 = image_tools.convert_images_to_video_b64(cam_high_images, fps=25)
        data = {
            "video_b64": video_b64
        }
        return jsonify({"status": "success", "message": "video capture finished", "data": data}), 200
    return jsonify({"status": "error", "message": "unknown action", "data":{}}), 400

if __name__ == '__main__':
    app.run(host='192.168.1.115', port=5000, debug=True)
