"""NaVILA's answers as motion of a legged robot: low-level steps of 25 cm or 15 degrees, run as velocity commands
closed on odometry.

The same code drives Go2 in Isaac Lab (a walking policy under velocity commands) and is meant for the real Go2,
whose built-in controller takes the same commands (Unitree SDK2 `Move(vx, vy, wz)`). Neither low level follows a
velocity exactly (Isaac Lab's Go2 policy turns slowly at small yaw rates and drifts about 9 degrees per 2 m), so a
step ends when the pose says so, not after a fixed time.

Habitat semantics are kept: an answer such as "move forward 75 cm" is 3 steps of 25 cm, and a frame is taken at
each step boundary. Consecutive steps of one answer run as one motion without stopping in between.

Obstacles: the student sees a camera 1.25 m up (NaVILA's training height), which misses the floor within about
1.25 m, and VLN-CE (where the student learned) slides the agent along whatever it walks into. Given the free
distance ahead (`Gap`, from a low depth camera on the robot), a forward step side-steps to the nearest gap when
something is close ahead, then walks on: a stand-in for the robot's own low-level avoidance (Go2's built-in LiDAR
avoidance, or NaVILA's height-map locomotion policy). Without that input the executor behaves as before.
Python 3.8 compatible, no dependencies.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3        # Habitat / VLN-CE action ids, as lightnav.vlnce.navila_actions
STEP_M = 0.25
STEP_RAD = math.radians(15.0)

Pose = Tuple[float, float, float]              # x, y (m), yaw (rad), world frame
# free distance straight ahead (m), and which side to step to if it comes to that (+ = left, - = right; None: no way
# round in view)
Gap = Tuple[float, Optional[float]]
Command = Tuple[float, float, float]           # vx, vy (m/s, body frame), wz (rad/s)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class MotionConfig:
    v_max: float = 0.5          # forward speed (m/s)
    v_min: float = 0.15         # never slower while a forward step is unfinished: the policy stalls near zero
    k_dist: float = 1.5         # forward speed per metre left
    k_heading: float = 2.0      # yaw rate per radian of heading error while walking
    w_hold_max: float = 0.6
    w_turn: float = 1.0         # yaw rate for turns: Isaac Lab's Go2 policy barely turns below about 0.5 rad/s
    pos_tol: float = 0.01       # a forward step counts as done this close to its end (m)
    yaw_tol: float = math.radians(2.0)
    step_timeout: float = 3.0   # seconds per step before the step is given up (blocked by a wall, say)
    stop_dist: float = 0.30     # side-step once the free distance ahead drops below this (m) ...
    clear_dist: float = 0.45    # ... and walk on once it is above this again
    v_side: float = 0.3         # side-step speed (m/s)
    avoid_max_s: float = 6.0    # side-stepping time per segment before the segment counts as blocked
    stall_s: float = 1.0        # a forward step that gains less than stall_m in stall_s is pressed against something
    stall_m: float = 0.05       #   the depth view missed (too close, too low) ...
    stall_side_s: float = 1.5   # ... and side-steps at least this long before trying ahead again


class Segment:
    """`k` identical low-level steps (`action` FORWARD, LEFT or RIGHT) from `pose`, run as one motion.

    Call `update(pose, dt)` every control tick: it returns the command for this tick (None once finished) and how
    many step boundaries were crossed since the previous call (a frame is due at each).
    """

    def __init__(self, action: int, k: int, pose: Pose, cfg: Optional[MotionConfig] = None):
        if action not in (FORWARD, LEFT, RIGHT):
            raise ValueError("action must be FORWARD, LEFT or RIGHT, got %r" % (action,))
        if k < 1:
            raise ValueError("k must be at least 1, got %r" % (k,))
        self.action, self.k, self.cfg = action, k, cfg or MotionConfig()
        self.x0, self.y0, self.yaw0 = pose
        self.prev_yaw, self.turned = pose[2], 0.0
        self.crossed, self.elapsed, self.blocked, self.finished = 0, 0.0, False, False
        self.avoiding, self.avoid_time, self.avoids = 0, 0.0, 0     # side-step direction (+1 left, -1 right, 0 none)
        self.avoid_start, self.avoid_min, self.stalls = 0.0, 0.0, 0
        self.ref = (0.0, 0.0)                                        # (elapsed, along) the stall check compares with

    def progress(self) -> float:
        """Steps done so far (fractional)."""
        if self.action == FORWARD:
            return self.along / STEP_M
        return self.turned * (1 if self.action == LEFT else -1) / STEP_RAD

    def update(self, pose: Pose, dt: float, gap: Optional[Gap] = None) -> Tuple[Optional[Command], int]:
        """`gap`: the free space ahead now (forward steps only); None turns obstacle avoidance off."""
        x, y, yaw = pose
        c = self.cfg
        self.turned += wrap(yaw - self.prev_yaw)
        self.prev_yaw = yaw
        self.along = (x - self.x0) * math.cos(self.yaw0) + (y - self.y0) * math.sin(self.yaw0)
        no_way = False
        if self.action == FORWARD and gap is not None:
            ahead, side = gap
            stalled = False
            if not self.avoiding and self.elapsed - self.ref[0] >= c.stall_s:
                stalled = self.along - self.ref[1] < c.stall_m
                self.ref = (self.elapsed, self.along)
            if not self.avoiding and (ahead < c.stop_dist or stalled):
                if side is None:
                    no_way = True                       # give the step up rather than push into the obstacle
                else:
                    self.avoiding = 1 if side > 0 else -1
                    self.avoids += 1
                    self.stalls += stalled
                    self.avoid_start, self.avoid_min = self.avoid_time, (c.stall_side_s if stalled else 0.0)
            elif (self.avoiding and ahead > c.clear_dist
                  and self.avoid_time - self.avoid_start >= self.avoid_min):
                self.avoiding = 0
                self.ref = (self.elapsed, self.along)
        if self.avoiding:
            self.avoid_time += dt
        else:
            self.elapsed += dt
        tol = c.pos_tol / STEP_M if self.action == FORWARD else c.yaw_tol / STEP_RAD
        done = min(self.k, int(math.floor(self.progress() + tol)))
        if done < self.k and (no_way or self.elapsed > c.step_timeout * self.k or self.avoid_time > c.avoid_max_s):
            done, self.blocked = self.k, True
        new, self.crossed = max(0, done - self.crossed), max(self.crossed, done)
        if self.crossed >= self.k:
            self.finished = True
            return None, new
        if self.action == FORWARD:
            wz = max(-c.w_hold_max, min(c.w_hold_max, c.k_heading * wrap(self.yaw0 - yaw)))
            if self.avoiding:
                return (0.0, self.avoiding * c.v_side, wz), new
            left_m = self.k * STEP_M - self.along
            vx = min(c.v_max, max(c.v_min, c.k_dist * left_m))
            return (vx, 0.0, wz), new
        return (0.0, 0.0, c.w_turn if self.action == LEFT else -c.w_turn), new
