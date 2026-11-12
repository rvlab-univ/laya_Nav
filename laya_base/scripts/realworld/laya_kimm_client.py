#!/usr/bin/env python3
"""ROS 2 client: drives the AI Worker in the KIMM Isaac Sim scene with LayaNav (laya_server.py).

Each control cycle, as in the Habitat evaluation loop:
  1. stop, take the front frame (head level)
  2. tilt the head 30 degrees down (head_joint1), take the look-down frame, tilt back
  3. send both to the server, then
       action STOP          -> episode ends
       action LEFT / RIGHT  -> turn 15 degrees in place
       path                 -> follow it with pure pursuit for up to --cycle_dist metres; on the way, every
                               --observe_dist metres a front frame goes to /observe (history only)
       arrived              -> next cycle re-decides
Every frame carries the /odom pose and the command executed since the previous frame (FORWARD / LEFT /
RIGHT / NONE), the v4 memory inputs; the command, not the measured move, so a blocked robot shows up as
FORWARD with no displacement.
At start the torso lift is lowered so the head camera sits at about 1.25 m, the height LayaNav was
trained with; both frames are centre-cropped from the ZED X Mini (105 deg) to 79 deg, like DualVLN.

Run with the system Python (ROS 2 Jazzy), with Isaac Sim (KIMM, playing), the swerve controller and
laya_server.py running:
    source /mnt/storage/dualVln/ros_env.sh
    python3 /mnt/storage/laya_work/isaac/laya_kimm_client.py --instruction "..."
"""
import argparse
import base64
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState

MODEL_FX = 386.5  # 79 deg HFOV at 640 x 480, the camera LayaNav (and DualVLN) were trained with
CAM_H, PITCH = 1.25, math.radians(30)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Client(Node):
    def __init__(self, cli):
        super().__init__("laya_kimm_client")
        self.cli = cli
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.rgb, self.rgb_seq, self.fx, self.odom, self.joints = None, 0, None, None, {}
        self.create_subscription(Image, cli.rgb_topic, self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, cli.info_topic, self.on_info, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self.on_joints, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.joint_pub = self.create_publisher(JointState, "/isaac_sim/joint_commands", 10)
        self.head_target = 0.0
        self.mode, self.path, self.goal, self.done = "IDLE", None, None, False

    # ---- ROS callbacks ----
    def on_info(self, msg):
        self.fx = float(msg.k[0])

    def on_rgb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        with self.lock:
            self.rgb, self.rgb_seq = img, self.rgb_seq + 1

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def on_joints(self, msg):
        with self.lock:
            self.joints.update(dict(zip(msg.name, msg.position)))

    # ---- commands ----
    def move(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.cmd_pub.publish(t)

    def publish_pose(self):
        """Torso lift and head targets; republished continuously so they hold."""
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = ["lift_joint", "head_joint1", "head_joint2"]
        m.position = [float(self.cli.lift), float(self.head_target), 0.0]
        self.joint_pub.publish(m)

    def wait_joint(self, name, target, tol, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                v = self.joints.get(name)
            if v is not None and abs(v - target) < tol:
                return True
            time.sleep(0.05)
        return False

    def fresh_frame(self, n_new=2, timeout=3.0):
        """A frame rendered after now (skip n_new - 1 frames that may still show the old pose)."""
        with self.lock:
            s0 = self.rgb_seq
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.rgb_seq >= s0 + n_new:
                    return self.rgb.copy()
            time.sleep(0.02)
        with self.lock:
            return self.rgb.copy()

    def crop(self, img):
        h, w = img.shape[:2]
        fx = self.fx if self.fx else 245.3 * w / 640
        cw = min(w, int(round(640 * fx / MODEL_FX)))
        ch = min(h, int(round(cw * 3 / 4)))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw], (640, 480), interpolation=cv2.INTER_LINEAR)

    # ---- control ----
    def control_step(self):
        with self.lock:
            odom, mode, path, goal = self.odom, self.mode, self.path, self.goal
        if odom is None or mode in ("IDLE", "DONE"):
            self.move(0.0, 0.0)
            return
        x, y, yaw = odom
        if mode == "TRAJ" and path is not None and len(path) > 1:
            d = np.linalg.norm(path - np.array([x, y]), axis=1)
            i = int(np.argmin(d))
            if i >= len(path) - 1 and d[-1] < 0.10:
                self.move(0.0, 0.0)
                return
            j = i
            while j < len(path) - 1 and np.linalg.norm(path[j] - np.array([x, y])) < self.cli.lookahead:
                j += 1
            alpha = wrap(math.atan2(path[j][1] - y, path[j][0] - x) - yaw)
            w = float(np.clip(1.5 * alpha, -self.cli.w_max, self.cli.w_max))
            v = 0.0 if abs(alpha) > math.radians(50) else self.cli.v_max * math.cos(alpha)
            self.move(v, w)
        elif mode == "TURN" and goal is not None:
            e = wrap(goal - yaw)
            self.move(0.0, float(np.clip(1.5 * e, -self.cli.w_max, self.cli.w_max)))
        else:
            self.move(0.0, 0.0)


def control_thread(node):
    k = 0
    while rclpy.ok() and not node.done:
        node.control_step()
        if k % 2 == 0:
            node.publish_pose()
        k += 1
        time.sleep(0.05)
    for _ in range(10):
        node.move(0.0, 0.0)
        time.sleep(0.05)


def to_pixel(xy):
    """Robot-frame floor point (x fwd, y left) -> pixel in the 30-degree look-down frame (flat floor)."""
    x, y = xy
    zc = x * math.cos(PITCH) + CAM_H * math.sin(PITCH)  # depth along the optical axis
    if zc < 0.05:
        return None
    yc = -x * math.sin(PITCH) + CAM_H * math.cos(PITCH)  # down in the image
    return int(320 + MODEL_FX * (-y) / zc), int(240 + MODEL_FX * yc / zc)


def visualize(front, down, r, step):
    f = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
    d = cv2.cvtColor(down, cv2.COLOR_RGB2BGR)
    if r.get("path"):
        pts = [to_pixel(p) for p in [(0.0, 0.0)] + r["path"]]
        pts = [p for p in pts if p is not None]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(d, a, b, (40, 220, 40), 3)
    if r.get("pixel_goal"):
        gx, gy = r["pixel_goal"]
        cv2.circle(d, (int(gx), int(gy)), 9, (40, 40, 230), 3)
    text = f"#{step} {r.get('kind')} {r.get('action', '')} {'decide' if r.get('decided') else ''}"
    out = np.concatenate([f, d], 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text + "   (left: front, right: look-down 30 deg)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def b64jpg(img):
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(jpg.tobytes()).decode()


def planning_thread(node):
    cli = node.cli
    os.makedirs(cli.out_dir, exist_ok=True)
    log = open(os.path.join(cli.out_dir, "log.tsv"), "w")
    writer = cv2.VideoWriter(os.path.join(cli.out_dir, "run.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2, (1280, 480))
    url = f"http://127.0.0.1:{cli.port}/step"
    observe_url = f"http://127.0.0.1:{cli.port}/observe"

    print("[client] waiting for camera, odom and joint states ...", flush=True)
    while rclpy.ok() and (node.rgb is None or node.odom is None or "lift_joint" not in node.joints):
        time.sleep(0.2)
    print(f"[client] lowering the torso to lift={cli.lift} (head camera at about 1.25 m)", flush=True)
    ok = node.wait_joint("lift_joint", cli.lift, 0.01, 20.0)
    print(f"[client] ready (lift reached: {ok}, camera fx={node.fx}). instruction: {cli.instruction}", flush=True)

    t_start, step, reset = time.time(), 0, True
    prev_action, n_observed = "NONE", 0  # command executed since the previous frame sent to the server
    while rclpy.ok() and time.time() - t_start < cli.max_seconds and step < cli.max_steps:
        with node.lock:
            node.mode = "IDLE"
        time.sleep(0.3)
        node.head_target = 0.0
        node.wait_joint("head_joint1", 0.0, 0.02, 3.0)
        front = node.crop(node.fresh_frame())
        node.head_target = PITCH
        node.wait_joint("head_joint1", PITCH, 0.02, 3.0)
        time.sleep(0.2)
        down = node.crop(node.fresh_frame())
        node.head_target = 0.0
        with node.lock:
            x, y, yaw = node.odom
        try:
            r = requests.post(url, json={"front_jpeg_b64": b64jpg(front), "down_jpeg_b64": b64jpg(down),
                                         "instruction": cli.instruction, "reset": reset, "pose": [x, y, yaw],
                                         "prev_action": prev_action}, timeout=60).json()
        except Exception as e:
            print(f"[client] server error: {e}", flush=True)
            time.sleep(1.0)
            continue
        reset = False

        kind = r.get("kind")
        if kind == "path":
            p = np.asarray(r["path"], dtype=np.float64)
            c, s = math.cos(yaw), math.sin(yaw)
            world = np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]], 1)
            node.wait_joint("head_joint1", 0.0, 0.05, 3.0)  # move with the head level, as the robot looks ahead
            with node.lock:
                node.path, node.mode = world, "TRAJ"
            t0, start = time.time(), np.array([x, y])
            last_obs, observed = start, 0
            while rclpy.ok() and time.time() - t0 < cli.cycle_time:
                with node.lock:
                    pos = np.array(node.odom[:2])
                if np.linalg.norm(pos - start) >= cli.cycle_dist or np.linalg.norm(pos - world[-1]) < 0.1:
                    break
                if cli.observe_dist > 0 and np.linalg.norm(pos - last_obs) >= cli.observe_dist:
                    img = node.crop(node.fresh_frame())  # head is level while moving
                    with node.lock:
                        ox, oy, oyaw = node.odom
                    try:
                        requests.post(observe_url, json={"front_jpeg_b64": b64jpg(img), "pose": [ox, oy, oyaw],
                                                         "prev_action": "FORWARD"}, timeout=10)
                        observed += 1
                    except Exception as e:
                        print(f"[client] observe error: {e}", flush=True)
                    last_obs = np.array([ox, oy])
                time.sleep(0.05)
            prev_action = "FORWARD"
            n_observed += observed
            text = f"path {r['path_length']:.2f} m -> end ({p[-1][0]:.2f}, {p[-1][1]:.2f}), +{observed} frames on the way"
        elif kind == "action":
            text = f"action {r['action']}"
            prev_action = r["action"] if r["action"] in ("LEFT", "RIGHT") else "NONE"
            if r["action"] in ("LEFT", "RIGHT"):
                target = wrap(yaw + (1 if r["action"] == "LEFT" else -1) * math.radians(15))
                with node.lock:
                    node.goal, node.mode = target, "TURN"
                t0 = time.time()
                while rclpy.ok() and time.time() - t0 < 5.0:
                    with node.lock:
                        e = abs(wrap(target - node.odom[2]))
                    if e < math.radians(2):
                        break
                    time.sleep(0.05)
        else:
            prev_action = "NONE"
            text = "arrived at the goal -> re-decide"

        elapsed = time.time() - t_start
        print(f"[client] step {step:3d} t={elapsed:5.1f}s pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}deg) {text} "
              f"(conf {r.get('confidence', '-')}, esc {r.get('escalate', '-')}, {r.get('time')}s)", flush=True)
        log.write(f"{step}\t{elapsed:.2f}\t{x:.3f}\t{y:.3f}\t{yaw:.3f}\t{kind}\t{r.get('action', '')}\t"
                  f"{r.get('path_length', '')}\t{r.get('confidence', '')}\t{r.get('escalate', '')}\n")
        log.flush()
        vis = visualize(front, down, r, step)
        writer.write(vis)
        cv2.imwrite(os.path.join(cli.out_dir, f"frame_{step:04d}.jpg"), vis)
        step += 1
        if kind == "action" and r["action"] == "STOP":
            print("[client] LayaNav decided STOP -> episode finished", flush=True)
            break

    with node.lock:
        node.mode = "DONE"
    writer.release()
    log.close()
    print(f"[client] finished: {step} cycles (+{n_observed} frames on the way), {time.time() - t_start:.0f}s. "
          f"Output: {cli.out_dir}", flush=True)
    time.sleep(0.5)
    node.done = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", required=True)
    p.add_argument("--out_dir", default="/mnt/storage/laya_work/outputs/isaac_kimm_" + time.strftime("%m%d_%H%M%S"))
    p.add_argument("--port", type=int, default=5802)
    p.add_argument("--rgb_topic", default="/zed_mini/rgb")
    p.add_argument("--info_topic", default="/zed_mini/camera_info")
    p.add_argument("--lift", type=float, default=-0.33, help="torso lift (m); -0.33 puts the head camera at ~1.25 m")
    p.add_argument("--v_max", type=float, default=0.25)
    p.add_argument("--w_max", type=float, default=0.6)
    p.add_argument("--lookahead", type=float, default=0.35)
    p.add_argument("--cycle_dist", type=float, default=0.5, help="metres to follow a plan before the next cycle")
    p.add_argument("--cycle_time", type=float, default=4.0, help="seconds at most per cycle")
    p.add_argument("--observe_dist", type=float, default=0.25,
                   help="metres between front frames sent on the way (v4 memory; 0 = off)")
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--max_seconds", type=float, default=600)
    cli = p.parse_args()

    rclpy.init()
    node = Client(cli)
    control = threading.Thread(target=control_thread, args=(node,), daemon=True)
    control.start()
    threading.Thread(target=planning_thread, args=(node,), daemon=True).start()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.done = True
        control.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
