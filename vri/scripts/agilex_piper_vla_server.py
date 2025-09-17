from flask import Flask, request, jsonify
from vri.scripts import runtime as _runtime
from vri.subscribers import video_display as _video_display
from vri.environments import piper_env as _piper_env 
from vri.agents import policy_agent as _policy_agent
from vri.policies import action_chunk_policy as _action_chunk_policy
from vri.policies import websocket_client_policy as _websocket_client_policy
from vri.utils.constants import CAM_HIGH, CAM_LEFT_WRIST, CAM_RIGHT_WRIST, CHUNK_SIZE
import logging
logging.getLogger().setLevel(logging.INFO)
import numpy as np

"""
conda create -n piper_run_env_v179 python=3.10.18
conda activate piper_run_env_v179
pip install flask==3.1.1
pip install imageio==2.37.0
pip install imageio-ffmpeg==0.6.0
pip install opencv-python==4.12.0.88
pip install Pillow
pip install imageio
pip install einops
pip install matplotlib==3.10.0
pip install dm_env==1.6
pip install websockets==14.1
pip install msgpack==1.1.0
export PYTHONPATH=/home/qluan/users/quebinbin/workspace/projects/vla_robots_interface:$PYTHONPATH
python vri/scripts/agilex_piper_vla_server.py
"""

app = Flask(__name__)
remote_model_ip = "192.168.1.187"
ws_client_base = _websocket_client_policy.WebsocketClientPolicy(host=remote_model_ip, port=8000)
ws_client_dict = {
    "base": ws_client_base
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


@app.route('/vla', methods=['POST', 'GET'])
def vla():
    data = request.json
    logging.info(f"received request: {data}")
    instruction = data.get('action')
    ws_client_policy = ws_client_dict['base']
    logging.info(f"instruction: {instruction}, uri: {ws_client_policy._uri}")
    metadata = ws_client_policy.get_server_metadata()
    print("pose:",metadata.get("reset_pose"))
    logging.info(f"Server metadata: {metadata}")
    
    runtime = _runtime.Runtime(
            environment=_piper_env.PiperEnvironment(reset_position=metadata.get("reset_pose", None), instruction=instruction, reset_type=3),
            agent=_policy_agent.PolicyAgent(
                policy=_action_chunk_policy.ActionChunkPolicy(
                    policy=ws_client_policy,
                    action_chunk_size=CHUNK_SIZE
                )
            ),
            # subscriber=_video_display.VideoDisplay([CAM_LEFT_WRIST, CAM_HIGH, CAM_RIGHT_WRIST]),
            subscriber=None,  # No video display in this case
            max_hz=5,
            num_episodes=1,
            max_episode_steps=10*500
        )
    action_status_type, video_b64 = runtime.run()
    data = {
            "action_status_type": action_status_type,
            "video_b64": video_b64
    }
    return jsonify({"status": "success", "message": "robot action complete", "data": data}), 200
    # try:
    #     runtime = _runtime.Runtime(
    #         environment=_piper_env.PiperEnvironment(reset_position=metadata.get("reset_pose", None), instruction=instruction, reset_type=3),
    #         agent=_policy_agent.PolicyAgent(
    #             policy=_action_chunk_policy.ActionChunkPolicy(
    #                 policy=ws_client_policy,
    #                 action_chunk_size=CHUNK_SIZE
    #             )
    #         ),
    #         # subscriber=_video_display.VideoDisplay([CAM_LEFT_WRIST, CAM_HIGH, CAM_RIGHT_WRIST]),
    #         subscriber=None,  # No video display in this case
    #         max_hz=7,
    #         num_episodes=1,
    #         max_episode_steps=10*500
    #     )
    #     action_status_type, video_b64 = runtime.run()
    #     data = {
    #         "action_status_type": action_status_type,
    #         "video_b64": video_b64
    #     }
    #     return jsonify({"status": "success", "message": "robot action complete", "data": data}), 200
    # except Exception as e:
    #     robot = runtime._environment._env.robot
    #     robot.send_action([0] * 14)
    #     for i in range(robot.config.follower_number):
    #             robot.follower_robot[i].DisconnectPort()

if __name__ == '__main__':
    app.run(host='192.168.0.210', port=5000, debug=True)
        
