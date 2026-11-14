"""The motion executor on an ideal unicycle and on a sluggish one (turns at half the commanded rate, drifts)."""
import math

import pytest

from lightnav.executor import FORWARD, LEFT, RIGHT, STEP_M, STEP_RAD, MotionConfig, Segment, wrap


def drive(seg, pose, dt=0.02, gain=1.0, drift=0.0, max_t=20.0, gap=None, path=None):
    """Integrate the commands (body-frame vx, vy); returns (final pose, tick index of each boundary).
    `gap(pose)` gives the free space ahead for obstacle avoidance; `path` collects every pose."""
    x, y, yaw = pose
    ticks, t = [], 0
    while t * dt < max_t:
        cmd, new = seg.update((x, y, yaw), dt, gap((x, y, yaw)) if gap else None)
        ticks += [t] * new
        if cmd is None:
            return (x, y, yaw), ticks
        vx, vy, wz = cmd
        yaw = wrap(yaw + (gain * wz + drift) * dt)
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        if path is not None:
            path.append((x, y))
        t += 1
    raise AssertionError("segment never finished")


@pytest.mark.parametrize("yaw0", [0.0, 2.5, -1.0])
def test_forward_three_steps_holds_heading(yaw0):
    seg = Segment(FORWARD, 3, (1.0, 2.0, yaw0))
    (x, y, yaw), ticks = drive(seg, (1.0, 2.0, yaw0), drift=0.05)
    along = (x - 1.0) * math.cos(yaw0) + (y - 2.0) * math.sin(yaw0)
    assert 3 * STEP_M - 0.011 <= along <= 3 * STEP_M + 0.03
    assert abs(wrap(yaw - yaw0)) < math.radians(3)
    assert len(ticks) == 3 and ticks == sorted(ticks) and not seg.blocked


@pytest.mark.parametrize("action,sign", [(LEFT, 1), (RIGHT, -1)])
def test_turns_end_on_the_measured_angle_even_when_slow(action, sign):
    seg = Segment(action, 2, (0.0, 0.0, 3.1))
    (_, _, yaw), ticks = drive(seg, (0.0, 0.0, 3.1), gain=0.5)        # across the +-pi seam too
    turned = wrap(yaw - 3.1) * sign
    assert 2 * STEP_RAD - math.radians(2.1) <= turned <= 2 * STEP_RAD + math.radians(2)
    assert len(ticks) == 2


def test_blocked_step_times_out_and_still_reports_every_boundary():
    seg = Segment(FORWARD, 2, (0.0, 0.0, 0.0), MotionConfig(step_timeout=0.5))
    crossed, cmd = 0, (0.0, 0.0, 0.0)
    for _ in range(100):
        cmd, new = seg.update((0.0, 0.0, 0.0), 0.02)                      # never moves
        crossed += new
        if cmd is None:
            break
    assert cmd is None and seg.blocked and crossed == 2


def test_rejects_stop_and_empty():
    with pytest.raises(ValueError):
        Segment(0, 1, (0, 0, 0))
    with pytest.raises(ValueError):
        Segment(FORWARD, 0, (0, 0, 0))


HALF_W = 0.2                                # robot half width used by the stand-in depth check


def box_gap(x0, x1, y0, y1):
    """Free space for a robot heading +x past the box [x0, x1] x [y0, y1]: what the drive script's depth check
    reports (distance ahead in the robot's corridor, nearest side offset with a clear corridor, 0.1 m steps)."""
    def blocked(x, yc):
        return x < x1 and yc + HALF_W > y0 and yc - HALF_W < y1

    def gap(pose):
        x, y, _ = pose
        ahead = x0 - x if blocked(x, y) else 3.0
        side = None
        if ahead < MotionConfig().stop_dist:
            for s in (0.1 * i for i in range(1, 9)):
                free = [d * s for d in (1, -1) if not blocked(x, y + d * s)]
                if free:
                    side = free[0]
                    break
        return ahead, side
    return gap, blocked


def test_forward_side_steps_round_an_obstacle_to_the_nearer_gap():
    gap, blocked = box_gap(0.45, 0.75, -0.2, 0.35)           # the gap on the right is nearer
    seg, path = Segment(FORWARD, 3, (0.0, 0.0, 0.0)), []
    (x, y, yaw), ticks = drive(seg, (0.0, 0.0, 0.0), gap=gap, path=path)
    along = x                                              # heading +x
    assert not seg.blocked and len(ticks) == 3 and along >= 3 * STEP_M - 0.011
    assert seg.avoids == 1 and seg.avoid_time > 0 and y < -0.39 and abs(yaw) < math.radians(3)
    assert not any(blocked(px, py) and px > 0.45 for px, py in path)   # never walked into the box


def test_forward_gives_up_at_once_when_there_is_no_way_round():
    def wall(pose):
        return 0.2 - pose[0], None
    seg = Segment(FORWARD, 2, (0.0, 0.0, 0.0))
    (x, _, _), ticks = drive(seg, (0.0, 0.0, 0.0), gap=wall)
    assert seg.blocked and len(ticks) == 2 and x < 0.15 and seg.elapsed < 0.5


def test_turns_ignore_the_gap():
    seg = Segment(LEFT, 1, (0.0, 0.0, 0.0))
    drive(seg, (0.0, 0.0, 0.0), gap=lambda pose: (0.05, None))
    assert not seg.blocked and seg.avoids == 0


def test_stalled_forward_side_steps_round_an_obstacle_the_depth_view_missed():
    """A wall at x = 0.3 for y > -0.4 that the depth check never reports: the step stalls, side-steps the way the
    gap points (right) and walks on."""
    seg, dt = Segment(FORWARD, 3, (0.0, 0.0, 0.0)), 0.02
    x = y = 0.0
    for _ in range(2000):
        cmd, _ = seg.update((x, y, 0.0), dt, (3.0, -0.1))
        if cmd is None:
            break
        vx, vy, _ = cmd
        nx, y = x + vx * dt, y + vy * dt
        x = nx if (nx <= 0.3 or y < -0.4) else x
    assert cmd is None and not seg.blocked and seg.stalls >= 1 and y < -0.4 and x >= 3 * STEP_M - 0.011
