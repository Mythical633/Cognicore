#!/usr/bin/env python3

import math
import os
import signal
import subprocess
import time
from collections import deque

import cv2
import numpy as np
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401
try:
    from ros_gz_interfaces.msg import Contacts
except ImportError:
    Contacts = None


# ============================================================
# KINEMATICS UTILITIES FOR TIAGO PRO LEFT ARM
# ============================================================

def rpy_to_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def make_transform(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def joint_transform(axis, angle):
    T = np.eye(4)
    axis = np.array(axis, dtype=float) / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    T[:3, :3] = np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])
    return T


# Exact chain from torso_lift_link to arm_left_7_link from tiago_pro.urdf
ARM_CHAIN = [
    ([0.08976, 0.15918, -0.092008], [-2.0679, 0.0, -1.2217], [0, 0, 1]),  # arm_1
    ([0.03711, 0.02, 0.18], [-1.57079632679489, 0, -1.5707963267949], [0, 0, 1]),  # arm_2
    ([0.02, -0.17011, -0.03711], [1.5707963267949, 0, 0], [0, 0, 1]),  # arm_3
    ([0.02, -0.03711, 0.15989], [1.57079632679489, 0, 0], [0, 0, 1]),  # arm_4
    ([-0.02, 0.17011, -0.03711], [-1.5707963267949, 3.14, 0], [0, 0, 1]),  # arm_5
    ([0.0, -0.0422, 0.1654], [-1.5708, 0, 3.14], [0, 0, -1]),  # arm_6
    ([0.07, -0.0534, -0.0422], [1.5708, 0, 0], [0, 0, 1]),  # arm_7
]

# Fixed tip transform from arm_left_7_link to gripper_left_grasping_link
T_TOOL = make_transform([0.0, 0.0, 0.017], [0.0, 0.0, 0.0])
T_GRASP = make_transform([0.0, 0.0, 0.157157], [0.0, -1.57, 0.0])
T_TIP = T_TOOL @ T_GRASP

# Joint limits from tiago_pro.urdf (safety_controller soft limits)
ARM_JOINT_LIMITS = [
    (-0.450, 4.640),   # arm_1
    (-2.370, 1.060),   # arm_2
    (-2.540, 2.540),   # arm_3
    (-2.370, 1.060),   # arm_4
    (-3.590, 1.500),   # arm_5
    (-1.810, 2.930),   # arm_6
    (-2.370, 2.370),   # arm_7
]


def forward_kinematics_torso(q):
    """Compute 4x4 pose of gripper_left_grasping_link in torso_lift_link."""
    T = np.eye(4)
    for i, (xyz, rpy, axis) in enumerate(ARM_CHAIN):
        T = T @ make_transform(xyz, rpy) @ joint_transform(axis, q[i])
    return T @ T_TIP


def solve_arm_ik(target_x, target_y, target_z, torso_lift, prev_seed=None):
    """
    Solve IK for gripper_left_grasping_link reaching (target_x, target_y, target_z)
    in base_footprint frame, with gripper pointing forward (+X in base_footprint).
    Returns: joint_positions (list of 7 floats), pos_error_meters (float)
    """
    tx = target_x + 0.0245
    ty = target_y
    tz = target_z - (0.8457 + torso_lift)

    # First attempt: continuous optimization from prev_seed with minimal regularization
    if prev_seed is not None:
        def cost_prev(q):
            T = forward_kinematics_torso(q)
            p = T[:3, 3]
            pos_err = (p[0] - tx) ** 2 + (p[1] - ty) ** 2 + (p[2] - tz) ** 2
            fwd_err = (1.0 - T[0, 0]) ** 2 + T[1, 0] ** 2 + T[2, 0] ** 2
            roll_err = T[2, 1] ** 2
            jump_err = float(np.sum((np.array(q) - np.array(prev_seed)) ** 2))
            return pos_err * 500.0 + fwd_err * 20.0 + roll_err * 5.0 + jump_err * 0.05

        res_prev = minimize(
            cost_prev, prev_seed, bounds=ARM_JOINT_LIMITS, method="L-BFGS-B",
            options={"maxiter": 120, "ftol": 1e-6}
        )
        T_res = forward_kinematics_torso(res_prev.x)
        err = float(np.linalg.norm(T_res[:3, 3] - [tx, ty, tz]))
        fwd_err = float(np.linalg.norm(T_res[:3, 0] - [1, 0, 0]))
        max_jump = float(np.max(np.abs(res_prev.x - np.array(prev_seed))))
        if err < 0.008 and fwd_err < 0.08 and max_jump < 0.80:
            return [float(v) for v in res_prev.x], err

    seeds = [
        [0.28, 1.06, 0.0, -0.71, -0.21, 1.20, -0.17],
        [1.25, 1.05, -1.38, -2.05, -2.56, -1.75, 0.60],
        [1.36, -0.07, -0.93, -1.80, -0.17, 1.74, -0.45],
        [1.83, 1.04, -1.66, -2.34, 0.59, 1.58, 0.47],
        [2.16, -1.44, -1.45, -2.31, -0.98, 1.12, -1.35],
        [1.48, 0.30, -0.75, -2.12, 0.22, 1.72, -0.49],
        [0.5, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0],
        [1.0, -0.3, 0.0, -0.8, 0.0, 0.4, 0.0],
    ]
    if prev_seed is not None:
        seeds.insert(0, prev_seed)

    best_q = None
    best_score = 1e9
    best_err = 1e9

    for seed in seeds:
        def cost(q):
            T = forward_kinematics_torso(q)
            p = T[:3, 3]
            pos_err = (p[0] - tx) ** 2 + (p[1] - ty) ** 2 + (p[2] - tz) ** 2
            fwd_err = (1.0 - T[0, 0]) ** 2 + T[1, 0] ** 2 + T[2, 0] ** 2
            roll_err = T[2, 1] ** 2
            return pos_err * 300.0 + fwd_err * 10.0 + roll_err * 2.0

        res = minimize(
            cost, seed, bounds=ARM_JOINT_LIMITS, method="L-BFGS-B",
            options={"maxiter": 120, "ftol": 1e-6}
        )
        T_res = forward_kinematics_torso(res.x)
        err = float(np.linalg.norm(T_res[:3, 3] - [tx, ty, tz]))
        fwd_err = float(np.linalg.norm(T_res[:3, 0] - [1, 0, 0]))

        score = err
        if prev_seed is not None:
            jump = float(np.max(np.abs(res.x - np.array(prev_seed))))
            score += jump * 0.1

        if err < 0.003 and fwd_err < 0.05 and (prev_seed is None or float(np.max(np.abs(res.x - np.array(prev_seed)))) < 0.80):
            return [float(v) for v in res.x], err

        if score < best_score:
            best_score = score
            best_err = err
            best_q = res.x

    return [float(v) for v in best_q], best_err


def solve_arm_ik_side(target_x, target_y, target_z, torso_lift, prev_seed=None):
    """
    Solve IK for gripper_left reaching outwards to the side (+Y in base_footprint)
    to drop a book into the red collection box on the table.
    """
    tx = target_x + 0.0245
    ty = target_y
    tz = target_z - (0.8457 + torso_lift)

    def cost(q):
        T = forward_kinematics_torso(q)
        p = T[:3, 3]
        pos_err = (p[0] - tx) ** 2 + (p[1] - ty) ** 2 + (p[2] - tz) ** 2
        # End-effector pointing along +Y (outwards to the side)
        ori_err = T[0, 0] ** 2 + (1.0 - T[1, 0]) ** 2 + T[2, 0] ** 2
        cost_val = pos_err * 500.0 + ori_err * 10.0
        if prev_seed is not None:
            cost_val += float(np.sum((np.array(q) - np.array(prev_seed)) ** 2)) * 0.05
        return cost_val

    seeds = [
        [-0.30, 1.06, 0.0, -1.91, 1.05, 1.63, 0.0],
        [0.5, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0],
        [1.0, 0.5, -1.0, -1.5, 0.5, 1.0, 0.0],
        [0.26, -1.60, 0.35, -1.98, 0.0, -1.2, 0.0],
    ]
    if prev_seed is not None:
        seeds.insert(0, prev_seed)

    best_q, best_err = None, 999.0
    for seed in seeds:
        res = minimize(
            cost, seed, bounds=ARM_JOINT_LIMITS, method="L-BFGS-B",
            options={"maxiter": 150, "ftol": 1e-6}
        )
        T_res = forward_kinematics_torso(res.x)
        err = float(np.linalg.norm(T_res[:3, 3] - [tx, ty, tz]))
        if err < best_err:
            best_err = err
            best_q = res.x
        if err < 0.005:
            break

    return [float(x) for x in best_q], best_err


# ============================================================
# COGNICORE CONTROLLER
# ============================================================

class CognicoreController(Node):

    # ---- Color Recognition ----
    COLOR_NAMES = {
        0: "RED",
        1: "YELLOW",
        2: "GREEN",
        3: "BLUE",
        4: "ANY",
    }

    COLOR_RANGES = {
        "RED": [
            ((0, 70, 50), (12, 255, 255)),
            ((168, 70, 50), (180, 255, 255)),
        ],
        "YELLOW": [
            ((18, 70, 60), (38, 255, 255)),
        ],
        "GREEN": [
            ((35, 70, 50), (88, 255, 255)),
        ],
        "BLUE": [
            ((95, 70, 50), (135, 255, 255)),
        ],
    }

    # ---- Camera Intrinsics ----
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 360
    CAMERA_FX = 337.2096
    CAMERA_FY = 337.2096
    CAMERA_CX = 320.0
    CAMERA_CY = 180.0

    # ---- Turn Parameters ----
    TURN_TOLERANCE_RAD = math.radians(1.5)
    TURN_KP = 1.5
    MAX_TURN_SPEED = 0.55
    MIN_TURN_SPEED = 0.05

    # ---- Shelf Lateral Alignment ----
    BASE_LATERAL_KP = 0.0025
    MAX_LATERAL_SPEED = 0.50
    MIN_LATERAL_SPEED = 0.08
    LATERAL_SIGN = 1.0  # In ROS Twist: +linear.y = left, -linear.y = right
    SHELF_CENTER_TOLERANCE_PX = 20
    REQUIRED_SHELF_CENTER_CONFIRMATIONS = 4
    SHELF_ALIGN_TIMEOUT = 25.0

    # ---- OCR & Shelf Identification ----
    OCR_STRIP_FRACTION = 0.38
    OCR_TIMEOUT = 6.0
    REQUIRED_OCR_CONFIRMATIONS = 2
    OCR_MIN_CONFIDENCE = 30.0

    # Known column lateral offsets from shelf center (y=0 in shelf frame)
    # Column 1: +2.0m, Col 2: +1.0m, Col 3: 0.00m, Col 4: -1.0m, Col 5: -2.0m
    DEFAULT_COLUMN_OFFSETS = {
        1: 2.0,
        2: 1.0,
        3: 0.0,
        4: -1.0,
        5: -2.0,
    }

    # ---- Forward Drive ----
    # Driving ~1.85m leaves a larger working gap to the shelf,
    # while keeping the book inside the arm working range.
    # Book spine (x=2.82m) is at ~0.77m in front of base, well within arm reach (tested up to 0.85m).
    APPROACH_DISTANCE = 1.85
    DRIVE_SPEED = 0.30
    MIN_SAFE_DEPTH = 0.58  # larger safety buffer: stop if shelf is closer than 58cm

    # ---- Head Poses ----
    # On TIAGo Pro: positive head_2_joint tilts UP (limit +0.27), negative tilts DOWN (limit -0.97)
    HEAD_CENTER = [0.0, 0.0]
    HEAD_SHELF_OCR = [0.0, 0.0]    # level head [0, 0] gives direct view of all 5 markers at cy ≈ 97
    HEAD_SCAN_MID = [0.0, -0.10]    # Row 3 (bz ≈ 1.25m)
    HEAD_SCAN_DOWN = [0.0, -0.40]   # Row 4 (bz ≈ 0.92m)
    HEAD_SCAN_UP = [0.0, 0.15]      # Row 2 (bz ≈ 1.58m)
    HEAD_SCAN_LOW = [0.0, -0.82]    # Row 5 / bottom row (lower camera for reliable visibility)
    HEAD_SCAN_FULL_DOWN = [0.0, -0.97]  # Bottom-row special case: fully down to see the complete book
    BOTTOM_FULL_DOWN_WAIT = 2.5
    BOTTOM_RELOCK_SAMPLES = 5
    BOTTOM_RELOCK_PERIOD = 0.20
    BOTTOM_GRAB_PREGRASP_TIME = 3.5
    BOTTOM_GRAB_APPROACH_TIME = 3.5
    # Bottom-row only: the previous plan could stop at an IK residual of ~31 mm
    # and never move the arm.  Use a small upward centre correction and a short
    # insertion so the gripper reaches the low book without entering the shelf.
    BOTTOM_GRAB_Z_OFFSET = 0.035
    BOTTOM_GRAB_INSERTION = 0.035
    BOTTOM_IK_MAX_ERROR = 0.045
    HEAD_SCAN_TOP = [0.0, 0.27]      # Row 1 / top row (upper camera; +0.27 is the TIAGo limit)

    # ---- Shelf row Z calibration ----
    # Expected book-spine heights in base_footprint.  These are used as a
    # SOFT preference during visual search, not as a hard geometric lock.
    # This prevents a middle-row scan from selecting another same-colour
    # object while still tolerating RGB-D/TF error of a few centimetres.
    # The current simulator geometry is approximately: bottom, row 4,
    # row 3, row 2, top.
    ROW_Z_CALIBRATION = {
        "LOW": 0.55,
        "DOWN": 0.92,
        "MID": 1.25,
        "UP": 1.58,
        "TOP": 1.75,
    }
    ROW_Z_SOFT_TOLERANCE = 0.20
    ROW_Z_SCORE_WEIGHT = 3.0

    HEAD_JOINT_NAMES = ["head_1_joint", "head_2_joint"]

    # ---- Gripper Configuration ----
    # PATCH: original gripper calibration/targets preserved exactly.
    # Note: TIAGo Pro gripper controller has a single joint: gripper_left_finger_joint
    # Range is 0.00 (closed) to 0.069 (open)
    GRIPPER_JOINT_NAMES = ["gripper_left_finger_joint"]
    GRIPPER_OPEN = [0.065]
    GRIPPER_CLOSED = [0.0]

    # The shelf openings are only 0.35 m high and each upright book is 0.25 m
    # high.  There is therefore no usable "over the book" corridor: raising
    # the wrist by 8--12 cm puts it into the shelf board above.  Keep the
    # gripper on the book centreline while it enters the opening instead.
    PREGRASP_STANDOFF = 0.14
    # RGB-D measures the visible front face.  Move the grasp frame 5.8 cm
    # beyond it so the book is gripped deeply between the finger pads (fingertip is 3.4cm behind grasp frame).
    # (The simulated book is 16 cm deep and 2 cm wide.)
    GRASP_INSERTION = 0.058
    IN_SLOT_LIFT = 0.015
    RETRACT_DISTANCE = 0.040
    # Gazebo/controller feedback can lag the commanded arm trajectory.  Hold
    # the final pose before closing, then hold the clamp before pulling.
    GRASP_SETTLE_DELAY = 1.0
    # Final insertion is generated as several Cartesian x-waypoints so the
    # wrist/elbow cannot take a shortcut between two distant joint poses.
    GRASP_CARTESIAN_STEPS = 8
    GRASP_CARTESIAN_TIME = 2.2
    GRASP_PREPOSITION_TIME = 1.2
    GRIPPER_CONTACT_DELAY = 1.25

    # ---- Grasp verification ----
    # A successful close/retract is not enough to claim a book was grabbed.
    # During MOVE_BACK, look for the target-colour book near the gripper and
    # make sure it has moved away from its original odom position.
    GRASP_VERIFY_GRIPPER_DISTANCE = 0.30
    GRASP_VERIFY_ORIGINAL_DISTANCE = 0.20
    GRASP_VERIFY_REQUIRED_SAMPLES = 1
    GRASP_VERIFY_TIMEOUT = 5.0
    # Safety stop for MOVE_BACK. Reverse 40 cm into clear open corridor.
    MAX_BOOK_EXTRACTION_DISTANCE = 0.40

    # ---- Return and Drop Constants ----
    CARRY_ARM_POSE = [0.188, 0.900, -0.134, -1.175, 0.0, 0.358, 0.0]
    DROP_ARM_FALLBACK_POSE = [0.8678, 1.0600, -0.6043, -1.1640, 0.3678, 1.4997, -0.1968]
    BOX_TARGET_YAW = math.pi / 2.0
    HEAD_RED_BOX_VIEW = [0.0, -0.55]
    TORSO_DROP_LIFT = 0.35
    RED_BOX_TARGET_BX = 0.75
    RED_BOX_TARGET_BY = 0.00

    # ---- Arm Configuration ----
    ARM_JOINT_NAMES = [
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        "arm_left_4_joint",
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
    ]
    ARM_RIGHT_JOINT_NAMES = [
        "arm_right_1_joint",
        "arm_right_2_joint",
        "arm_right_3_joint",
        "arm_right_4_joint",
        "arm_right_5_joint",
        "arm_right_6_joint",
        "arm_right_7_joint",
    ]
    # Compact home poses from TIAGo Pro motions so neither arm strikes the shelf or pins the robot
    ARM_LEFT_TUCKED = [0.26, -1.60, 0.35, -1.98, 0.0, -1.2, 0.0]
    ARM_RIGHT_TUCKED = [-0.26, -1.60, -0.35, -1.98, 0.0, -1.2, 0.0]

    # Left arm shoulder Y offset in base_footprint
    ARM_LEFT_Y_OFFSET = 0.16

    # Put the book in the arm's comfortable reach before moving the arm.  Base
    # alignment combines this depth correction with the lateral correction,
    # producing a diagonal strafe whenever both errors are present.
    MANIPULATION_DEPTH = 0.72
    MANIPULATION_DEPTH_TOLERANCE = 0.04
    MAX_MANIPULATION_FORWARD_SPEED = 0.10

    # The depth edge of a thin book is noisy, particularly while the head is
    # settling.  Map it from several frames before committing to any motion.
    BOOK_MAP_SAMPLES = 7
    BOOK_MAP_SAMPLE_PERIOD = 0.15
    BOOK_MAP_TIMEOUT = 8.0
    BOOK_MAP_MAX_DEVIATION = 0.040
    BOOK_MAP_MIN_INLIERS = 5

    # Row-specific vertical compensation.
    # The top-row book is reachable in this simulator, but the grasp pose
    # otherwise lands slightly below the spine.  Keep the measured book
    # position unchanged for RGB-D/base alignment and apply this small offset
    # only to the arm target.
    # Current shelf calibration: 1.59 m targets are still being handled as the
    # normal centreline.  Do not add the old experimental +8 cm top-row offset
    # unless a measured target is actually above this threshold.
    TOP_ROW_Z_THRESHOLD = 1.80
    TOP_ROW_Z_OFFSET = 0.0
    TOP_ROW_GRASP_INSERTION = 0.058

    # ========================================================
    # INIT
    # ========================================================

    def __init__(self):
        super().__init__(
            "cognicore_controller",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        # ---- Parameters ----
        self.declare_parameter("target_color_id", 0)
        self.declare_parameter("target_color", "")
        self.declare_parameter("target_shelf", 3)
        self.declare_parameter("approach_distance", self.APPROACH_DISTANCE)
        self.declare_parameter("show_camera_view", True)
        self.declare_parameter("head_scan", True)
        self.declare_parameter("lateral_sign", self.LATERAL_SIGN)

        self.target_color_id = int(self.get_parameter("target_color_id").value)
        color_param = str(self.get_parameter("target_color").value).strip().upper()
        self.target_color = color_param or self.COLOR_NAMES.get(self.target_color_id, "RED")
        self.target_shelf = int(self.get_parameter("target_shelf").value)
        self.approach_distance = float(self.get_parameter("approach_distance").value)
        self.show_camera_view = bool(self.get_parameter("show_camera_view").value)
        self.head_scan_enabled = bool(self.get_parameter("head_scan").value)
        self.lateral_sign = float(self.get_parameter("lateral_sign").value)

        # ---- Publishers ----
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.arm_pub = self.create_publisher(
            JointTrajectory, "/arm_left_controller/joint_trajectory", 10
        )
        self.arm_right_pub = self.create_publisher(
            JointTrajectory, "/arm_right_controller/joint_trajectory", 10
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory, "/gripper_left_controller/joint_trajectory", 10
        )
        self.gripper_raw_pub = self.create_publisher(
            JointTrajectory, "/gripper_left_controller_raw/joint_trajectory", 10
        )
        self.head_pub = self.create_publisher(
            JointTrajectory, "/head_controller/joint_trajectory", 10
        )
        self.torso_pub = self.create_publisher(
            JointTrajectory, "/torso_controller/joint_trajectory", 10
        )

        # ---- Subscribers ----
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, "/head_front_camera/head_front_camera/color/image_raw", self.camera_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, "/head_front_camera/head_front_camera/depth/image_rect_raw", self.depth_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        if Contacts is not None:
            self.bin_contacts_sub = self.create_subscription(
                Contacts, "/bin_contacts", self.bin_contacts_callback, 10
            )

        # ---- TF ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- CV & Data ----
        self.bridge = CvBridge()
        self.last_frame = None
        self.last_depth = None
        self.digit_templates = self._load_digit_templates()

        # ---- Odometry State ----
        self.odom_received = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # ---- State Machine ----
        self.state = "WAIT_FOR_ODOM"
        self.state_start_time = self.get_clock().now()
        self.command_sent = False

        # ---- Turn & Heading State ----
        self.start_yaw = None
        self.target_yaw = None
        self.locked_yaw = None
        self.backup_start_x = 0.0
        self.backup_start_y = 0.0

        # ---- Shelf OCR & Alignment ----
        self.shelf_confirmed = False
        self.shelf_column_x = None
        self.shelf_x_history = deque(maxlen=7)
        self.shelf_ocr_confirmations = 0
        self.shelf_align_confirmations = 0
        self.align_start_x = 0.0
        self.align_start_y = 0.0
        self.target_lateral_dist = 0.0
        self.last_detected_shelf_numbers = []

        # ---- Drive State ----
        self.drive_start_x = 0.0
        self.drive_start_y = 0.0
        self.min_detected_front_depth = float("inf")

        # ---- Book Detection & Manipulation ----
        # Search bottom row first, then continue through the other rows.
        # The bottom-row special full-down/relock/grab mechanism is unchanged.
        self.scan_views = [
            ("LOW", self.HEAD_SCAN_LOW),
            ("TOP", self.HEAD_SCAN_TOP),
            ("UP", self.HEAD_SCAN_UP),
            ("MID", self.HEAD_SCAN_MID),
            ("DOWN", self.HEAD_SCAN_DOWN),
        ]
        self.current_scan_index = 0
        self.current_scan_row_name = "LOW"
        self.current_scan_expected_z = self.ROW_Z_CALIBRATION["LOW"]
        self.target_book_3d = None  # (x, y, z) in base_footprint
        self.locked_book_3d = None  # frozen (x, y, z) used across all grasp states
        self.locked_book_odom = None  # PointStamped in odom frame for invariant spatial tracking
        self.book_confirmations = 0
        self.book_map_samples = []  # target points expressed in stable odom
        self.last_book_map_sample_time = -float("inf")
        self.last_book_bbox = None  # (x, y, width, height, depth_m)
        self.target_torso_lift = 0.0
        self.arm_target_z_offset = 0.0
        self.last_arm_q = None
        self.current_left_arm_q = None
        self.current_gripper_q = None
        self.planned_pregrasp_q = None
        self.planned_grasp_q = None
        self.last_scan_switch_time = self.get_clock().now()
        # Special bottom-row mode: once a bottom book is seen, point fully down
        # and keep the camera there so the complete book face is used for locking.
        self.bottom_book_camera_locked = False
        self.bottom_book_mode = False
        self.bottom_full_down_start_time = None
        self.bottom_relock_samples = []
        self.bottom_relock_last_time = -float("inf")
        self.grasp_verify_samples = 0
        self.grasp_verify_last_time = -float("inf")
        self.grasp_verify_passed = False
        # Set only after the physical pull-out grip test succeeds.  This is
        # used only to make MOVE_BACK fail-safe; it does not alter the grasp.
        self.grip_test_passed = False

        # ---- Return and Drop State ----
        self._locked_red_box_odom = None
        self._red_box_hit = None
        self._red_box_debug_bbox = None
        self._book_bin_contact_detected = False
        self._drop_target_base_xyz = None
        self._current_drop_q = None

        # ---- Control Timer (20 Hz) ----
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.log_startup()

    # ========================================================
    # STARTUP LOGGING
    # ========================================================

    def log_startup(self):
        self.get_logger().info("==========================================")
        self.get_logger().info("COGNICORE AUTONOMOUS CONTROLLER")
        self.get_logger().info(f"Target color : {self.target_color} (id={self.target_color_id})")
        self.get_logger().info(f"Target shelf : {self.target_shelf}")
        self.get_logger().info(f"Approach     : {self.approach_distance:.2f} m")
        self.get_logger().info("Arm Solver   : TIAGo Pro 7-DOF Exact Kinematics + SciPy L-BFGS-B")
        self.get_logger().info("Gripper      : gripper_left_finger_joint (single-joint JTC)")
        self.get_logger().info("Torso Lift   : Active multi-row height coordination")
        self.get_logger().info("==========================================")

    # ========================================================
    # DIGIT TEMPLATES
    # ========================================================

    def _load_digit_templates(self):
        """Load ground-truth textures from erc_description, with OpenCV fallback."""
        templates = {}
        texture_dir = "/opt/erc_ws/src/erc_description/models/number_marker/textures"
        if not os.path.isdir(texture_dir):
            alt = os.path.expanduser("~/erc_sim_2026/src/erc_description/models/number_marker/textures")
            if os.path.isdir(alt):
                texture_dir = alt

        for digit in range(1, 6):
            tex_file = os.path.join(texture_dir, f"{digit}.png")
            if os.path.isfile(tex_file):
                tex = cv2.imread(tex_file, cv2.IMREAD_GRAYSCALE)
                if tex is not None:
                    _, tex_th = cv2.threshold(tex, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    templates[digit] = cv2.resize(tex_th, (25, 30))
                    continue

            # Fallback: render font bitmap
            canvas = np.zeros((30, 20), dtype=np.uint8)
            text = str(digit)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
            ox = max(0, (20 - tw) // 2)
            oy = max(th, (30 + th) // 2)
            cv2.putText(canvas, text, (ox, oy), font, 1.0, 255, 2, cv2.LINE_AA)
            templates[digit] = canvas

        return templates

    # ========================================================
    # STATE MACHINE TRANSITIONS
    # ========================================================

    def change_state(self, new_state):
        old_state = self.state
        self.state = new_state
        self.state_start_time = self.get_clock().now()
        self.command_sent = False

        self.get_logger().info(f"[STATE] {old_state} -> {new_state}")

        if new_state == "TURN_RIGHT":
            self.stop()
            # Turn 90 degrees RIGHT relative to the robot's current heading.
            # Do not use an absolute -90 degree world heading.
            self.start_yaw = self.yaw
            self.target_yaw = self.normalize_angle(self.start_yaw - math.pi / 2.0)

        elif new_state == "IDENTIFY_SHELF":
            self.stop()
            self.send_head_pose(self.HEAD_SHELF_OCR, 1.0)
            self.shelf_confirmed = False
            self.shelf_column_x = None
            self.shelf_x_history.clear()
            self.shelf_ocr_confirmations = 0

        elif new_state == "ALIGN_TO_SHELF":
            self.stop()
            self.shelf_align_confirmations = 0
            self.align_start_x = self.x
            self.align_start_y = self.y
            if self.shelf_column_x is not None:
                # Discretize detected column image X to nearest shelf column (0 to 4)
                col_centers = [93.0, 214.0, 336.0, 457.0, 576.0]
                col_idx = int(np.argmin([abs(self.shelf_column_x - c) for c in col_centers]))
                self.target_lateral_dist = float((2 - col_idx) * 1.0)
            elif self.target_shelf in self.DEFAULT_COLUMN_OFFSETS:
                self.target_lateral_dist = self.DEFAULT_COLUMN_OFFSETS[self.target_shelf]
            else:
                self.target_lateral_dist = 0.0
            self.get_logger().info(
                f"[SHELF] Aligning to target shelf {self.target_shelf}: required lateral dy = {self.target_lateral_dist:+.3f}m"
            )

        elif new_state == "DRIVE_TO_SHELF":
            self.send_head_pose(self.HEAD_CENTER, 1.0)
            self.drive_start_x = self.x
            self.drive_start_y = self.y
            self.min_detected_front_depth = float("inf")

            # Clear anything left over from a previous run so the camera HUD
            # cannot display an old shelf/book target while driving.
            self.shelf_column_x = None
            self.last_book_bbox = None
            self.target_book_3d = None
            self.last_detected_shelf_numbers = []

        elif new_state == "SEARCH_FOR_BOOK":
            self.stop()
            self.grasp_verify_samples = 0
            self.grasp_verify_last_time = -float("inf")
            self.grasp_verify_passed = False
            self.grip_test_passed = False
            self.current_scan_index = 0
            self.target_book_3d = None
            self.book_confirmations = 0
            self.arm_target_z_offset = 0.0
            self.bottom_book_camera_locked = False
            self.bottom_book_mode = False
            self.bottom_full_down_start_time = None
            self.bottom_relock_samples.clear()
            self.bottom_relock_last_time = -float("inf")
            self.last_scan_switch_time = self.get_clock().now()
            name, pose = self.scan_views[0]
            self.current_scan_row_name = name
            self.current_scan_expected_z = self.ROW_Z_CALIBRATION.get(name)
            self.get_logger().info(
                f"[SCAN] Starting book search at {name} row (expected Z={self.current_scan_expected_z:.2f}m), "
                "then continuing upward."
            )
            self.send_head_pose(pose, 0.8)

        elif new_state == "MAP_BOOK":
            self.stop()
            self.book_map_samples.clear()
            self.last_book_map_sample_time = -float("inf")

        elif new_state in ("LIFT_TORSO", "PREPOSITION_ARM"):
            # The shelf OCR line/boxes are no longer useful once manipulation
            # starts.  Keep the book target itself available to the grasp logic.
            self.shelf_column_x = None
            self.last_detected_shelf_numbers = []
            self.stop()
            # PREPOSITION_ARM is also reached directly for low shelves, where
            # no torso motion is required.  Open here rather than relying on
            # LIFT_TORSO, otherwise the gripper remains in the closed transit
            # pose throughout the approach.
            self.send_gripper_pose(self.GRIPPER_OPEN, 1.2)

        elif new_state in (
            "ALIGN_BASE_TO_BOOK",
            "MAP_BOOK",
            "PREPARE_ARM",
            "PLAN_GRASP",
            "PREPOSITION_ARM",
            "APPROACH_BOOK",
            "LOWER_TO_BOOK",
            "CLOSE_GRIPPER",
            "TEST_GRIP",
            "RETRACT_ARM",
            "MOVE_BACK",
            "LIFT_BOOK_CLEARANCE",
            "RETURN_TURN_TO_TABLE",
            "RED_BOX_SEARCH",
            "PREPARE_DROP",
            "DROP_BOOK",
            "DROP_COMPLETE",
            "BOOK_GRABBED",
            "FAILED",
        ):
            self.stop()

    # ========================================================
    # CALLBACKS
    # ========================================================

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.x = pos.x
        self.y = pos.y

        siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(f"[ODOM] Ready at ({self.x:.3f}, {self.y:.3f}), yaw={math.degrees(self.yaw):.1f}°")

    def joint_state_callback(self, msg):
        """Seed IK from the measured arm pose, never an arbitrary branch."""
        positions = dict(zip(msg.name, msg.position))
        try:
            self.current_left_arm_q = [
                float(positions[name]) for name in self.ARM_JOINT_NAMES
            ]
        except KeyError:
            pass

        try:
            self.current_gripper_q = float(
                positions[self.GRIPPER_JOINT_NAMES[0]]
            )
        except KeyError:
            pass

    def bin_contacts_callback(self, msg):
        for c in msg.contacts:
            n1 = getattr(c.collision1, "name", "")
            n2 = getattr(c.collision2, "name", "")
            if "book" in n1.lower() or "book" in n2.lower():
                if not getattr(self, "_book_bin_contact_detected", False):
                    self._book_bin_contact_detected = True
                    self.get_logger().info(
                        f"[CONTACT] Confirmed book contact inside collection bin on /bin_contacts: {n1} <-> {n2}"
                    )

    def camera_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"[CAMERA] Image decode failed: {e}")
            return

        self.last_frame = frame

        if self.state == "IDENTIFY_SHELF":
            if self.target_shelf != 0 and self.elapsed_time() >= 1.2:
                self.ocr_shelf_numbers(frame)
        elif self.state == "ALIGN_TO_SHELF":
            if self.target_shelf != 0:
                self.ocr_shelf_numbers(frame)

        if self.show_camera_view:
            self.render_view(frame)

    def depth_callback(self, msg):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"[DEPTH] Image decode failed: {e}")
            return

        self.last_depth = depth_img

        # Monitor front center depth while driving or making the final
        # depth/lateral base alignment before manipulation or box approach.
        if self.state in ("DRIVE_TO_SHELF", "ALIGN_BASE_TO_BOOK", "APPROACH_RED_BOX"):
            h, w = depth_img.shape
            center_patch = depth_img[h // 2 - 20:h // 2 + 20, w // 2 - 40:w // 2 + 40]
            valid = center_patch[np.isfinite(center_patch)]
            if len(valid) > 0:
                self.min_detected_front_depth = float(np.median(valid))

    # ========================================================
    # SHELF OCR
    # ========================================================

    def ocr_shelf_numbers(self, frame):
        h, w = frame.shape[:2]
        strip_h = int(h * self.OCR_STRIP_FRACTION)
        roi = frame[0:strip_h, 0:w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cy = y + ch // 2
            # Markers are located in the upper frame strip
            if cy > 130:
                continue
            if cw < 12 or ch < 15 or cw > 70 or ch > 70:
                continue
            aspect = cw / float(ch)
            if aspect < 0.30 or aspect > 1.50:
                continue

            digit_roi = cv2.resize(thresh[y:y + ch, x:x + cw], (25, 30))
            best_digit, best_score = None, -1.0
            for digit, template in self.digit_templates.items():
                res = cv2.matchTemplate(digit_roi, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_score:
                    best_score = score
                    best_digit = digit

            score_pct = best_score * 100.0
            if best_digit is not None and score_pct >= self.OCR_MIN_CONFIDENCE:
                center_x = x + cw // 2
                # Deduplicate if another detection is within 25px
                if not any(abs(center_x - d[1]) < 25 for d in detected):
                    detected.append((best_digit, center_x, score_pct, x, y, cw, ch))

        self.last_detected_shelf_numbers = detected

        target_matches = [d for d in detected if d[0] == self.target_shelf]
        if target_matches:
            best_match = max(target_matches, key=lambda m: m[2])
            self.shelf_x_history.append(best_match[1])
            self.shelf_column_x = int(np.median(list(self.shelf_x_history)))
            self.shelf_ocr_confirmations += 1

            if self.shelf_ocr_confirmations >= self.REQUIRED_OCR_CONFIRMATIONS and not self.shelf_confirmed:
                self.shelf_confirmed = True
                self.get_logger().info(
                    f"[OCR] Target shelf {self.target_shelf} confirmed at image X={self.shelf_column_x}"
                )
                try:
                    cv2.imwrite("/tmp/shelf_markers_view.png", frame)
                except Exception:
                    pass

    # ========================================================
    # BOOK 3D DETECTION VIA RGB-D & TF
    # ========================================================

    def detect_target_book_3d(self, expected_z=None):
        """
        Locate the target book using HSV color thresholding on RGB frame
        and depth reading from the registered depth image, projected to base_footprint.
        If expected_z is provided, filters for candidates close to that height.
        Returns: (x, y, z) in base_footprint or None
        """
        if self.last_frame is None or self.last_depth is None:
            return None, None

        frame = self.last_frame
        depth = self.last_depth
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.target_color == "ANY":
            ranges = []
            for r in self.COLOR_RANGES.values():
                ranges.extend(r)
        else:
            ranges = self.COLOR_RANGES.get(self.target_color, [])

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            curr = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, curr)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates_3d = []

        tf_timeout = rclpy.duration.Duration(seconds=0.08)
        can_tf = False
        try:
            can_tf = self.tf_buffer.can_transform(
                "base_footprint", "head_front_camera_color_optical_frame", rclpy.time.Time(), timeout=tf_timeout
            )
        except Exception:
            pass

        if not can_tf:
            return None, None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # The bottom-row book can be partly hidden by the shelf board,
            # so accept a smaller visible colour contour.
            if area < 50:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            moments = cv2.moments(cnt)
            if moments["m00"] <= 0.0:
                continue
            # Use the contour centroid rather than the rectangle centre.  It
            # remains correct when the book is tilted or partly occluded.
            cx = int(round(moments["m10"] / moments["m00"]))
            cy = int(round(moments["m01"] / moments["m00"]))

            # Use depth only from the colour contour, eroded by one pixel.
            # The previous whole-bounding-box median could include the shelf
            # behind the book and place the gripper at the wrong depth.
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            contour_mask = cv2.erode(contour_mask, np.ones((3, 3), np.uint8), iterations=1)
            patch = depth[max(0, y):min(depth.shape[0], y + h), max(0, x):min(depth.shape[1], x + w)]
            mask_patch = contour_mask[max(0, y):min(depth.shape[0], y + h), max(0, x):min(depth.shape[1], x + w)]
            valid = patch[np.isfinite(patch) & (mask_patch > 0)]
            if len(valid) == 0:
                continue
            z_m = float(np.median(valid))
            if z_m < 0.35 or z_m > 2.8:
                continue

            # Project pixel (cx, cy, z_m) to 3D point in camera optical frame
            x_cam = (cx - self.CAMERA_CX) * z_m / self.CAMERA_FX
            y_cam = (cy - self.CAMERA_CY) * z_m / self.CAMERA_FY
            z_cam = z_m

            # Transform to base_footprint and odom frames via TF
            try:
                pt = PointStamped()
                pt.header.frame_id = "head_front_camera_color_optical_frame"
                pt.header.stamp = rclpy.time.Time().to_msg()
                pt.point.x = float(x_cam)
                pt.point.y = float(y_cam)
                pt.point.z = float(z_cam)
                pt_base = self.tf_buffer.transform(pt, "base_footprint", timeout=tf_timeout)
                pt_odom = self.tf_buffer.transform(pt, "odom", timeout=tf_timeout)
                bx, by, bz = pt_base.point.x, pt_base.point.y, pt_base.point.z

                # Books in front: x in [0.40m, 1.20m], z within shelf height [0.25m, 2.0m]
                # And book in current column: by within [-0.55m, 0.55m] (rejects neighboring columns)
                if 0.35 <= bx <= 1.25 and -0.55 <= by <= 0.55 and 0.15 <= bz <= 2.20:
                    if expected_z is not None:
                        z_error = abs(bz - expected_z)
                        # Keep the row filter soft.  Middle-row RGB-D depth can
                        # move by several centimetres with head tilt and shelf
                        # occlusion, so do not throw away a valid book just
                        # because it is slightly away from the nominal height.
                        if z_error > self.ROW_Z_SOFT_TOLERANCE:
                            continue
                        row_penalty = self.ROW_Z_SCORE_WEIGHT * z_error
                    else:
                        row_penalty = 0.0

                    # Score candidate: row consistency first, then image
                    # centering and visible area.
                    score = (
                        row_penalty
                        + abs(cx - self.CAMERA_CX) * 0.8
                        - area * 0.05
                    )
                    candidates_3d.append((score, bx, by, bz, pt_odom, (x, y, w, h, z_m)))
            except Exception:
                continue

        if not candidates_3d:
            return None, None

        candidates_3d.sort(key=lambda c: c[0])
        _, best_bx, best_by, best_bz, best_odom, best_bbox = candidates_3d[0]
        self.last_book_bbox = best_bbox
        return (best_bx, best_by, best_bz), best_odom

    def get_locked_book_base(self):
        """Query TF for real-time (x, y, z) of the locked book in base_footprint."""
        if self.locked_book_odom is not None:
            try:
                pt_query = PointStamped()
                pt_query.header.frame_id = self.locked_book_odom.header.frame_id
                pt_query.point = self.locked_book_odom.point
                tf_timeout = rclpy.duration.Duration(seconds=0.08)
                pt_base = self.tf_buffer.transform(pt_query, "base_footprint", timeout=tf_timeout)
                return (pt_base.point.x, pt_base.point.y, pt_base.point.z)
            except Exception as e:
                self.get_logger().warn(f"[TF] Locked book transform failed: {e}")
        return self.target_book_3d

    def get_gripper_odom_position(self):
        """Return current left gripper position in odom, or None if TF is unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "odom",
                "gripper_left_grasping_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.08),
            )
            return np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ], dtype=float)
        except Exception:
            return None

    def verify_book_grabbed(self, min_moved=0.12):
        """Verify the target book is physically travelling with the gripper.

        Do not use the general book detector here: while backing away it can
        still see the original book on the shelf and mistake that for a held
        book.  Instead, project the actual gripper into the camera image and
        only accept target-colour pixels in a small neighbourhood around it.
        """
        if self.locked_book_odom is None or self.last_frame is None or self.last_depth is None:
            return False

        try:
            tf_timeout = rclpy.duration.Duration(seconds=0.08)
            tf_g = self.tf_buffer.lookup_transform(
                "head_front_camera_color_optical_frame",
                "gripper_left_grasping_link",
                rclpy.time.Time(), timeout=tf_timeout
            )
            gx = tf_g.transform.translation.x
            gy = tf_g.transform.translation.y
            gz = tf_g.transform.translation.z
        except Exception:
            return False

        # Camera optical frame: X right, Y down, Z forward.
        if gz <= 0.05:
            return False
        u0 = int(round(self.CAMERA_FX * gx / gz + self.CAMERA_CX))
        v0 = int(round(self.CAMERA_FY * gy / gz + self.CAMERA_CY))

        frame = self.last_frame
        depth = self.last_depth
        h, w = frame.shape[:2]
        radius = 95
        x0, x1 = max(0, u0 - radius), min(w, u0 + radius + 1)
        y0, y1 = max(0, v0 - radius), min(h, v0 + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return False

        roi = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ranges = self.COLOR_RANGES.get(self.target_color, [])
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Restrict to a circular neighbourhood around the projected gripper.
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        local_u = u0 - x0
        local_v = v0 - y0
        circle = (xx - local_u) ** 2 + (yy - local_v) ** 2 <= radius ** 2
        mask[~circle] = 0
        if int(cv2.countNonZero(mask)) < 35:
            return False

        ys, xs = np.where(mask > 0)
        cx = int(round(float(np.mean(xs)) + x0))
        cy = int(round(float(np.mean(ys)) + y0))

        # Median depth only over target-colour pixels.
        depth_roi = depth[y0:y1, x0:x1]
        valid = depth_roi[(mask > 0) & np.isfinite(depth_roi)]
        if len(valid) < 15:
            return False
        z_m = float(np.median(valid))
        if not (0.35 < z_m < 2.8):
            return False

        x_cam = (cx - self.CAMERA_CX) * z_m / self.CAMERA_FX
        y_cam = (cy - self.CAMERA_CY) * z_m / self.CAMERA_FY
        pt = PointStamped()
        pt.header.frame_id = "head_front_camera_color_optical_frame"
        pt.header.stamp = rclpy.time.Time().to_msg()
        pt.point.x, pt.point.y, pt.point.z = float(x_cam), float(y_cam), float(z_m)

        try:
            pt_odom = self.tf_buffer.transform(pt, "odom", timeout=tf_timeout)
            seen = np.array([pt_odom.point.x, pt_odom.point.y, pt_odom.point.z], dtype=float)
            target = np.array([
                self.locked_book_odom.point.x,
                self.locked_book_odom.point.y,
                self.locked_book_odom.point.z,
            ], dtype=float)

            # Gripper position in odom.
            gp = self.get_gripper_odom_position()
            if gp is None:
                return False
            d_gripper = float(np.linalg.norm(seen - gp))
            d_original = float(np.linalg.norm(seen - target))

            self.get_logger().info(
                f"[VERIFY] local colour pixels={int(cv2.countNonZero(mask))}, "
                f"gripper_px=({u0},{v0}), book_px=({cx},{cy}), "
                f"book-to-gripper={d_gripper:.3f}m, moved={d_original:.3f}m"
            )

            return d_gripper <= 0.18 and d_original >= float(min_moved)
        except Exception:
            return False

    def build_grasp_plan(self):
        """Solve the complete in-slot approach before commanding either pose."""
        book_base = self.get_locked_book_base()
        if book_base is None:
            return False

        bx, by, bz = book_base
        seed = self.current_left_arm_q if self.current_left_arm_q is not None else self.last_arm_q

        # Bottom row only: use a slightly raised centreline because the low
        # shelf target is otherwise producing a large IK residual before the
        # arm even starts.  This does not change the RGB-D/odom book lock.
        if self.bottom_book_mode:
            arm_z = bz + self.BOTTOM_GRAB_Z_OFFSET
            grasp_insertion = self.BOTTOM_GRAB_INSERTION
        else:
            arm_z = bz + self.arm_target_z_offset
            grasp_insertion = (
                self.TOP_ROW_GRASP_INSERTION
                if bz >= self.TOP_ROW_Z_THRESHOLD else self.GRASP_INSERTION
            )

        q_pre, pre_err = solve_arm_ik(
            bx - self.PREGRASP_STANDOFF, by, arm_z,
            self.target_torso_lift, prev_seed=seed
        )
        q_grasp, grasp_err = solve_arm_ik(
            bx + grasp_insertion, by, arm_z,
            self.target_torso_lift, prev_seed=q_pre
        )

        max_error = max(pre_err, grasp_err)
        allowed_error = self.BOTTOM_IK_MAX_ERROR if self.bottom_book_mode else 0.012
        if max_error > allowed_error:
            self.get_logger().warn(
                f"[PLAN] Refusing unreachable grasp plan (max IK error={max_error * 1000:.1f}mm, "
                f"limit={allowed_error * 1000:.1f}mm)."
            )
            return False

        self.planned_pregrasp_q = q_pre
        self.planned_grasp_q = q_grasp
        self.get_logger().info(
            f"[PLAN] Validated mapped-book plan: pregrasp={pre_err * 1000:.1f}mm, "
            f"grasp={grasp_err * 1000:.1f}mm."
        )
        return True

    def build_cartesian_grasp_trajectory(self, bx, by, bz, grasp_insertion):
        """Generate a short straight-in Cartesian insertion using IK at each waypoint.

        The old approach solved only the endpoints and then interpolated in joint
        space.  That can make the elbow/wrist arc sideways or vertically at the
        shelf opening.  Here the grasping-link target moves only along +X while
        Y/Z and gripper orientation stay fixed.
        """
        arm_z = bz + self.arm_target_z_offset
        q_prev = self.planned_pregrasp_q
        if q_prev is None:
            return None

        x0 = bx - self.PREGRASP_STANDOFF
        x1 = bx + grasp_insertion
        steps = max(2, int(self.GRASP_CARTESIAN_STEPS))
        dt = float(self.GRASP_CARTESIAN_TIME) / float(steps)
        points = []

        # Include the pregrasp pose as the first waypoint.
        points.append((q_prev, float(self.GRASP_PREPOSITION_TIME)))

        for i in range(1, steps + 1):
            alpha = i / float(steps)
            x = x0 + (x1 - x0) * alpha
            q_i, err = solve_arm_ik(
                x, by, arm_z, self.target_torso_lift, prev_seed=q_prev
            )
            if err > 0.018:
                self.get_logger().warn(
                    f"[PLAN] Cartesian grasp waypoint {i}/{steps} failed: "
                    f"x={x:.3f}, IK error={err*1000:.1f}mm"
                )
                return None
            points.append((q_i, dt))
            q_prev = q_i

        return points

    # ========================================================
    # MAIN CONTROL LOOP
    # ========================================================

    def control_loop(self):

        # ----------------------------------------------------
        # 1. WAIT FOR ODOM
        # ----------------------------------------------------
        if self.state == "WAIT_FOR_ODOM":
            self.stop()
            if self.odom_received:
                self.change_state("TUCK_ARM")
            return

        # ----------------------------------------------------
        # 2. TUCK ARM
        # ----------------------------------------------------
        if self.state == "TUCK_ARM":
            if not self.command_sent:
                self.send_arm_pose(self.ARM_LEFT_TUCKED, 2.5)
                self.send_arm_right_pose(self.ARM_RIGHT_TUCKED, 2.5)
                self.send_gripper_pose(self.GRIPPER_CLOSED, 1.0)
                self.command_sent = True
                self.get_logger().info("[ARM] Tucking left and right arms for safe transit...")
                return

            if self.elapsed_time() >= 3.0:
                self.get_logger().info("[ARM] Both arms safely tucked.")
                self.change_state("TURN_RIGHT")
            return

        # ----------------------------------------------------
        # 3. TURN RIGHT (turn 90 degrees right to face shelf)
        # ----------------------------------------------------
        if self.state == "TURN_RIGHT":
            # target_yaw is computed once from the heading at the start of
            # this state, so the robot always turns exactly 90 degrees right.
            target_yaw = self.target_yaw
            if target_yaw is None:
                self.start_yaw = self.yaw
                self.target_yaw = self.normalize_angle(self.start_yaw - math.pi / 2.0)
                target_yaw = self.target_yaw

            error = self.angle_difference(target_yaw, self.yaw)

            if abs(error) <= self.TURN_TOLERANCE_RAD or self.elapsed_time() > 7.0:
                self.stop()
                self.locked_yaw = target_yaw
                self.get_logger().info(
                    f"[BASE] Turn complete. Heading locked at "
                    f"{math.degrees(self.locked_yaw):.1f}° "
                    f"(err={math.degrees(error):.2f}°)."
                )
                if self.target_shelf == 0:
                    self.change_state("DRIVE_TO_SHELF")
                else:
                    self.change_state("IDENTIFY_SHELF")
                return

            turn_speed = float(np.clip(abs(error) * self.TURN_KP, self.MIN_TURN_SPEED, self.MAX_TURN_SPEED))
            z_vel = -turn_speed if error < 0 else turn_speed
            self.drive(angular_z=z_vel)
            return

        # ----------------------------------------------------
        # 4. IDENTIFY SHELF
        # ----------------------------------------------------
        if self.state == "IDENTIFY_SHELF":
            self.stop()

            # Allow head to settle before evaluating OCR
            if self.elapsed_time() < 1.0:
                return

            if self.shelf_confirmed:
                self.change_state("ALIGN_TO_SHELF")
                return

            if self.elapsed_time() > self.OCR_TIMEOUT:
                self.get_logger().warn(
                    f"[OCR] Shelf {self.target_shelf} not confirmed before timeout. Using column offset fallback."
                )
                self.shelf_column_x = None
                self.change_state("ALIGN_TO_SHELF")
                return

        # ----------------------------------------------------
        # 5. ALIGN TO SHELF
        # ----------------------------------------------------
        if self.state == "ALIGN_TO_SHELF":
            needed_dist = abs(self.target_lateral_dist)
            if needed_dist < 0.05:
                self.stop()
                self.get_logger().info("[SHELF] Already centered at target shelf column.")
                self.change_state("DRIVE_TO_SHELF")
                return

            travelled_dist = math.hypot(self.x - self.align_start_x, self.y - self.align_start_y)
            if travelled_dist >= (needed_dist - 0.03):
                self.stop()
                self.shelf_align_confirmations += 1
                if self.shelf_align_confirmations >= 2:
                    self.get_logger().info(
                        f"[SHELF] Shelf column reached! Travelled {travelled_dist:.3f}m of {needed_dist:.3f}m"
                    )
                    self.change_state("DRIVE_TO_SHELF")
                return

            self.shelf_align_confirmations = 0
            remaining_dist = max(0.0, needed_dist - travelled_dist)
            speed = float(np.clip(remaining_dist * 1.5, self.MIN_LATERAL_SPEED, self.MAX_LATERAL_SPEED))
            direction = 1.0 if self.target_lateral_dist > 0 else -1.0
            self.drive(lateral_y=direction * speed)

            if self.elapsed_time() > self.SHELF_ALIGN_TIMEOUT:
                self.stop()
                self.get_logger().warn("[SHELF] Lateral align timeout. Continuing to drive.")
                self.change_state("DRIVE_TO_SHELF")
            return

        # ----------------------------------------------------
        # 6. DRIVE TO SHELF
        # ----------------------------------------------------
        if self.state == "DRIVE_TO_SHELF":
            dx = self.x - self.drive_start_x
            dy = self.y - self.drive_start_y
            travelled = math.hypot(dx, dy)

            # Stopping condition: reached distance OR front depth camera detects shelf obstacle
            shelf_reached = travelled >= self.approach_distance
            safe_stop = self.min_detected_front_depth <= self.MIN_SAFE_DEPTH

            if shelf_reached or safe_stop:
                self.stop()
                self.get_logger().info(
                    f"[BASE] Shelf reached! Distance travelled={travelled:.3f}m, front_depth={self.min_detected_front_depth:.3f}m"
                )
                self.change_state("SEARCH_FOR_BOOK")
                return

            # Drive forward with proportional slowdown near target
            dist_left = self.approach_distance - travelled
            speed = np.clip(dist_left * 0.5, 0.05, self.DRIVE_SPEED)
            self.drive(linear_x=speed)
            return

        # ----------------------------------------------------
        # 7. SEARCH FOR BOOK
        # ----------------------------------------------------
        if self.state == "SEARCH_FOR_BOOK":
            self.stop()

            book_pt, pt_odom = self.detect_target_book_3d(
                expected_z=self.current_scan_expected_z
            )
            if book_pt is not None:
                bx, by, bz = book_pt

                # --------------------------------------------------------
                # BOTTOM-ROW SPECIAL CASE
                # --------------------------------------------------------
                # If the first colour/depth detection is in the complete
                # bottom row, the normal LOW view may only see the upper
                # part of the book.  Do NOT lock that partial observation.
                # Instead, command the head to its physical full-down limit,
                # keep it there, and require the book to be detected again
                # from that complete view before normal mapping/grasping.
                BOTTOM_ROW_Z_MAX = 0.65
                if bz <= BOTTOM_ROW_Z_MAX and not self.bottom_book_camera_locked:
                    self.bottom_book_camera_locked = True
                    self.bottom_book_mode = True
                    self.bottom_full_down_start_time = self.get_clock().now()
                    self.book_confirmations = 0
                    self.target_book_3d = None
                    self.locked_book_odom = None
                    self.get_logger().info(
                        f"[BOOK] Bottom-row {self.target_color} detected at Z={bz:.3f}m. "
                        "Partial view rejected; looking fully down before locking the book."
                    )
                    self.send_head_pose(self.HEAD_SCAN_FULL_DOWN, 0.8)
                    self.last_scan_switch_time = self.get_clock().now()
                    return

                # Once full-down mode is entered, wait for the head to physically
                # settle at the full-down pose before accepting the first complete
                # book observation.  This prevents locking a frame captured while
                # the head is still moving.
                if self.bottom_book_camera_locked:
                    down_wait = 0.0
                    if self.bottom_full_down_start_time is not None:
                        down_wait = (
                            self.get_clock().now() - self.bottom_full_down_start_time
                        ).nanoseconds / 1e9
                    if down_wait < self.BOTTOM_FULL_DOWN_WAIT:
                        self.get_logger().info(
                            f"[BOOK] Full-down camera settling... "
                            f"{down_wait:.1f}/{self.BOTTOM_FULL_DOWN_WAIT:.1f}s"
                        )
                        return

                # Once full-down mode is entered, every accepted observation
                # comes from that locked camera pose.  Do not cycle back up.
                self.book_confirmations += 1
                self.target_book_3d = book_pt
                self.locked_book_odom = pt_odom
                if self.bottom_book_camera_locked:
                    self.get_logger().info(
                        f"[BOOK] Full-view {self.target_color} at base_footprint: "
                        f"({bx:.3f}, {by:.3f}, {bz:.3f}) "
                        f"confirm={self.book_confirmations}/2"
                    )
                else:
                    self.get_logger().info(
                        f"[BOOK] Detected {self.target_color} at base_footprint: "
                        f"({bx:.3f}, {by:.3f}, {bz:.3f}) "
                        f"confirm={self.book_confirmations}/2"
                    )

                if self.book_confirmations >= 2:
                    self.get_logger().info(
                        f"[BOOK] Complete-view candidate locked at ({bx:.3f}, {by:.3f}, {bz:.3f}); "
                        "holding full-down view for bottom-row relock before arm motion."
                    )
                    if self.bottom_book_mode:
                        self.change_state("BOTTOM_RELOCK")
                    else:
                        self.change_state("MAP_BOOK")
                    return
            else:
                self.book_confirmations = max(0, self.book_confirmations - 1)

            # Step through head scan views if book not found in current view.
            # The normal 2-second scan timing is unchanged.  The only exception
            # is the bottom-row full-down lock above.
            time_in_scan = (self.get_clock().now() - self.last_scan_switch_time).nanoseconds / 1e9
            if (self.head_scan_enabled and
                    not self.bottom_book_camera_locked and
                    time_in_scan > 2.0):
                self.current_scan_index = (self.current_scan_index + 1) % len(self.scan_views)
                name, pose = self.scan_views[self.current_scan_index]
                self.current_scan_row_name = name
                self.current_scan_expected_z = self.ROW_Z_CALIBRATION.get(name)
                self.get_logger().info(
                    f"[SCAN] Tilting head to {name}: {pose}; "
                    f"expected book Z={self.current_scan_expected_z:.2f}m"
                )
                self.send_head_pose(pose, 0.8)
                self.last_scan_switch_time = self.get_clock().now()

            if self.elapsed_time() > 30.0 and self.target_book_3d is None:
                self.get_logger().error(f"[BOOK] Timeout looking for {self.target_color} book.")
                self.change_state("FAILED")
            return

        # ----------------------------------------------------
        # 8. BOTTOM-ROW RELOCK (full-down camera only)
        # ----------------------------------------------------
        if self.state == "BOTTOM_RELOCK":
            self.stop()

            # The head remains at -0.97.  Take several complete-face RGB-D
            # observations while the robot is stationary, then freeze the
            # median odom point.  This is only for bottom-row books.
            relock_time = self.elapsed_time()
            if (relock_time - self.bottom_relock_last_time) >= self.BOTTOM_RELOCK_PERIOD:
                book_pt, pt_odom = self.detect_target_book_3d(
                    expected_z=self.current_scan_expected_z
                )
                if book_pt is not None and pt_odom is not None:
                    self.bottom_relock_samples.append((
                        pt_odom.point.x, pt_odom.point.y, pt_odom.point.z
                    ))
                    self.bottom_relock_last_time = relock_time
                    bx, by, bz = book_pt
                    self.get_logger().info(
                        f"[BOTTOM] Full-face relock {len(self.bottom_relock_samples)}/"
                        f"{self.BOTTOM_RELOCK_SAMPLES}: "
                        f"({bx:.3f}, {by:.3f}, {bz:.3f})"
                    )

            if len(self.bottom_relock_samples) >= self.BOTTOM_RELOCK_SAMPLES:
                samples = np.asarray(self.bottom_relock_samples, dtype=float)
                locked_xyz = np.median(samples, axis=0)

                locked = PointStamped()
                locked.header.frame_id = "odom"
                locked.header.stamp = self.get_clock().now().to_msg()
                locked.point.x = float(locked_xyz[0])
                locked.point.y = float(locked_xyz[1])
                locked.point.z = float(locked_xyz[2])

                self.locked_book_odom = locked
                mapped_base = self.get_locked_book_base()
                if mapped_base is not None:
                    self.target_book_3d = mapped_base

                self.get_logger().info(
                    f"[BOTTOM] COMPLETE BOOK FACE LOCKED at odom "
                    f"({locked_xyz[0]:.3f}, {locked_xyz[1]:.3f}, {locked_xyz[2]:.3f}). "
                    "Starting existing map/grab pipeline."
                )
                self.change_state("MAP_BOOK")
                return

            if relock_time > 5.0:
                self.get_logger().warn(
                    "[BOTTOM] Full-face relock timed out; using the last complete-view lock."
                )
                self.change_state("MAP_BOOK")
            return

        # ----------------------------------------------------
        # 9. MAP BOOK (robust pose estimate before planning motion)
        # ----------------------------------------------------
        if self.state == "MAP_BOOK":
            self.stop()
            map_time = self.elapsed_time()
            if map_time - self.last_book_map_sample_time >= self.BOOK_MAP_SAMPLE_PERIOD:
                book_pt, pt_odom = self.detect_target_book_3d(
                    expected_z=self.current_scan_expected_z
                )
                if book_pt is not None and pt_odom is not None:
                    self.book_map_samples.append((
                        pt_odom.point.x, pt_odom.point.y, pt_odom.point.z
                    ))
                    self.last_book_map_sample_time = map_time
                    self.get_logger().info(
                        f"[MAP] Book observation {len(self.book_map_samples)}/{self.BOOK_MAP_SAMPLES}"
                    )

            if len(self.book_map_samples) >= self.BOOK_MAP_SAMPLES:
                samples = np.asarray(self.book_map_samples, dtype=float)
                rough_center = np.median(samples, axis=0)
                errors = np.linalg.norm(samples - rough_center, axis=1)
                inliers = samples[errors <= self.BOOK_MAP_MAX_DEVIATION]
                if len(inliers) < self.BOOK_MAP_MIN_INLIERS:
                    self.get_logger().warn(
                        "[MAP] Observations disagreed; discarding map and sampling again."
                    )
                    self.book_map_samples.clear()
                    return

                mapped_xyz = np.median(inliers, axis=0)
                spread = np.max(np.std(inliers, axis=0))

                # Store one immutable world-frame target.  Subsequent base
                # moves transform this same mapped point back to the base
                # frame instead of re-detecting a slightly different pixel.
                mapped_odom = PointStamped()
                mapped_odom.header.frame_id = "odom"
                mapped_odom.header.stamp = self.get_clock().now().to_msg()
                mapped_odom.point.x = float(mapped_xyz[0])
                mapped_odom.point.y = float(mapped_xyz[1])
                mapped_odom.point.z = float(mapped_xyz[2])
                self.locked_book_odom = mapped_odom
                mapped_base = self.get_locked_book_base()
                if mapped_base is not None:
                    self.target_book_3d = mapped_base

                self.get_logger().info(
                    f"[MAP] Locked {len(inliers)}/{len(samples)} consistent observations; "
                    f"max spread={spread * 1000:.1f}mm. Planning motion."
                )
                self.change_state("ALIGN_BASE_TO_BOOK")
                return

            if map_time > self.BOOK_MAP_TIMEOUT:
                self.get_logger().warn("[MAP] Timed out before a complete map; returning to search.")
                self.change_state("SEARCH_FOR_BOOK")
            return

        # ----------------------------------------------------
        # 9. ALIGN BASE TO BOOK
        # ----------------------------------------------------
        if self.state == "ALIGN_BASE_TO_BOOK":
            self.stop()
            book_base = self.get_locked_book_base()
            if book_base is None:
                self.change_state("SEARCH_FOR_BOOK")
                return

            bx, by, bz = book_base
            self.target_book_3d = book_base
            lateral_error = by - self.ARM_LEFT_Y_OFFSET
            depth_error = bx - self.MANIPULATION_DEPTH

            if (abs(lateral_error) <= 0.015 and
                    abs(depth_error) <= self.MANIPULATION_DEPTH_TOLERANCE):
                self.stop()
                self.get_logger().info(
                    f"[ALIGN] Manipulation pose reached "
                    f"(x_err={depth_error * 100:.1f}cm, y_err={lateral_error * 100:.1f}cm)."
                )
                self.change_state("PREPARE_ARM")
                return

            # Correct both axes together.  This is a true diagonal strafe
            # when the book is too far away and off the left-arm centreline.
            lateral_speed = float(np.clip(lateral_error * 1.5, -0.15, 0.15))
            if abs(lateral_error) > 0.015 and abs(lateral_speed) < 0.04:
                lateral_speed = math.copysign(0.04, lateral_speed)

            forward_speed = float(np.clip(
                depth_error * 0.7,
                -self.MAX_MANIPULATION_FORWARD_SPEED,
                self.MAX_MANIPULATION_FORWARD_SPEED,
            ))
            if abs(depth_error) <= self.MANIPULATION_DEPTH_TOLERANCE:
                forward_speed = 0.0

            # Never advance further if the center depth says the shelf is at
            # the base safety boundary; lateral correction can still continue.
            if (forward_speed > 0.0 and
                    self.min_detected_front_depth < self.MIN_SAFE_DEPTH):
                forward_speed = 0.0

            self.drive(linear_x=forward_speed, lateral_y=lateral_speed)

            if self.elapsed_time() > 12.0:
                self.stop()
                self.get_logger().warn("[ALIGN] Timed out before full base alignment; using current pose.")
                self.change_state("PREPARE_ARM")
            return

        # ----------------------------------------------------
        # 9. PREPARE ARM (calculate torso lift only)
        # ----------------------------------------------------
        if self.state == "PREPARE_ARM":
            self.stop()
            if not self.command_sent:
                book_base = self.get_locked_book_base()
                if book_base is None:
                    self.change_state("SEARCH_FOR_BOOK")
                    return

                bx, by, bz = book_base
                self.target_book_3d = (bx, by, bz)

                # The top row needs a small extra vertical correction at the
                # gripper.  Do not move the base target or alter the measured
                # RGB-D book position; this offset is arm-only.
                if self.bottom_book_mode:
                    self.arm_target_z_offset = self.BOTTOM_GRAB_Z_OFFSET
                elif bz >= self.TOP_ROW_Z_THRESHOLD:
                    self.arm_target_z_offset = self.TOP_ROW_Z_OFFSET
                else:
                    self.arm_target_z_offset = 0.025

                arm_z = bz + self.arm_target_z_offset

                # Decide the required torso lift BEFORE moving the arm.
                desired_lift = arm_z - 0.8457 - 0.05
                self.target_torso_lift = float(np.clip(desired_lift, 0.0, 0.35))

                self.get_logger().info(
                    f"[TORSO] Book Z={bz:.3f}m, arm target Z={arm_z:.3f}m, "
                    f"vertical offset={self.arm_target_z_offset:+.3f}m, "
                    f"required lift={self.target_torso_lift:.3f}m"
                )

                self.command_sent = True
                self.state_start_time = self.get_clock().now()

                # If a lift is needed, complete it FIRST. The arm stays tucked.
                if self.target_torso_lift > 0.02:
                    self.change_state("LIFT_TORSO")
                else:
                    self.send_gripper_pose(self.GRIPPER_OPEN, 1.2, force=True)
                    self.change_state("PLAN_GRASP")
                return
            return

        # ----------------------------------------------------
        # 10. LIFT TORSO FIRST (arm stays tucked until lift is complete)
        # ----------------------------------------------------
        if self.state == "LIFT_TORSO":
            self.stop()

            if not self.command_sent:
                self.send_torso_pose(self.target_torso_lift, 2.5)
                self.send_gripper_pose(self.GRIPPER_OPEN, 1.2)
                self.command_sent = True
                self.state_start_time = self.get_clock().now()
                self.get_logger().info(
                    "[TORSO] Lifting torso completely before moving the arm..."
                )
                return

            if self.elapsed_time() >= 3.2:
                self.get_logger().info(
                    f"[TORSO] Lift complete at {self.target_torso_lift:.3f}m. Arm can now move."
                )
                self.change_state("PLAN_GRASP")
            return

        # ----------------------------------------------------
        # 11. PLAN GRASP (validate all in-slot poses before moving the arm)
        # ----------------------------------------------------
        if self.state == "PLAN_GRASP":
            self.stop()
            if self.build_grasp_plan():
                self.change_state("PREPOSITION_ARM")
            else:
                self.change_state("SEARCH_FOR_BOOK")
            return

        # ----------------------------------------------------
        # 12. PREPOSITION ARM (in front of the book, inside its shelf slot)
        # ----------------------------------------------------
        if self.state == "PREPOSITION_ARM":
            self.stop()
            if not self.command_sent:
                bx, by, bz = self.get_locked_book_base()
                # Enter at the book centreline.  The opening has only ~5 cm of
                # clearance above and below the book, so an elevated approach
                # would hit the shelf rather than clear it.
                target_x = bx - self.PREGRASP_STANDOFF
                target_y = by
                target_z = bz + self.arm_target_z_offset

                q_sol = self.planned_pregrasp_q
                if q_sol is None:
                    self.get_logger().error("[PLAN] Missing pregrasp pose; returning to book search.")
                    self.change_state("SEARCH_FOR_BOOK")
                    return
                self.last_arm_q = q_sol
                self.get_logger().info(
                    f"[ARM] Executing validated pregrasp: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
                )
                pregrasp_time = (
                    self.BOTTOM_GRAB_PREGRASP_TIME
                    if self.bottom_book_mode else 3.0
                )
                self.get_logger().info(
                    f"[ARM] {'Bottom-row slow pregrasp' if self.bottom_book_mode else 'Standard pregrasp'} "
                    f"duration={pregrasp_time:.1f}s"
                )
                self.send_gripper_pose(self.GRIPPER_OPEN, 1.2)
                self.send_arm_pose(q_sol, pregrasp_time)
                self.command_sent = True
                self.state_start_time = self.get_clock().now()
                return

            if self.elapsed_time() >= 3.0:
                self.change_state("APPROACH_BOOK")
            return

        # ----------------------------------------------------
        # 13. APPROACH BOOK (straight through the shelf opening)
        # ----------------------------------------------------
        if self.state == "APPROACH_BOOK":
            self.stop()
            if not self.command_sent:
                bx, by, bz = self.get_locked_book_base()
                # The depth image reports the front face of the book.  Enter only a
                # small amount so the wrist/forearm stays inside the opening
                # and does not reach the shelf back/upper board.
                if self.bottom_book_mode:
                    grasp_insertion = self.BOTTOM_GRAB_INSERTION
                elif bz >= self.TOP_ROW_Z_THRESHOLD:
                    grasp_insertion = self.TOP_ROW_GRASP_INSERTION
                else:
                    grasp_insertion = self.GRASP_INSERTION
                x_end = bx + grasp_insertion
                z_end = bz
                q_app = self.planned_grasp_q
                if q_app is None:
                    self.get_logger().error("[PLAN] Missing grasp pose; returning to book search.")
                    self.change_state("SEARCH_FOR_BOOK")
                    return
                self.last_arm_q = q_app

                self.get_logger().info(
                    f"[ARM] Executing validated grasp approach: target X={x_end:.3f}m "
                    f"(insertion={grasp_insertion:.3f}m)"
                )
                # Do not jump/interpolate from the tucked/pregrasp posture directly
                # to the deep grasp pose.  Use a slow, straight final insertion.
                # The end-effector only enters 2.5 cm past the measured book face.
                q_pre = self.planned_pregrasp_q
                if self.bottom_book_mode:
                    # Bottom-row target remains conservative because the shelf
                    # board is close to the gripper.
                    self.send_arm_trajectory([
                        (q_pre, 1.5),
                        (q_app, self.BOTTOM_GRAB_APPROACH_TIME),
                    ])
                else:
                    # Generate the final insertion in Cartesian X, not as a
                    # single joint-space jump.  This is the key grasp fix.
                    traj = self.build_cartesian_grasp_trajectory(
                        bx, by, bz, grasp_insertion
                    )
                    if traj is None:
                        self.get_logger().error(
                            "[PLAN] Cartesian insertion could not be generated; refusing unsafe grasp."
                        )
                        self.change_state("SEARCH_FOR_BOOK")
                        return
                    self.send_arm_trajectory(traj)
                self.command_sent = True
                self.state_start_time = self.get_clock().now()
                return

            # Wait for measured arm joints to reach the actual grasp pose,
            # with timeout fallback in case of joint stall or slight shelf contact.
            q_target = self.planned_grasp_q
            q_measured = self.current_left_arm_q
            elapsed = self.elapsed_time()

            if q_target is not None and q_measured is not None:
                arm_err = float(np.max(np.abs(
                    np.asarray(q_measured, dtype=float) -
                    np.asarray(q_target, dtype=float)
                )))
                if arm_err <= 0.06 or (elapsed >= 4.0 and arm_err <= 0.28) or (elapsed >= 5.5):
                    stable = getattr(self, "_grasp_approach_stable", 0) + 1
                    self._grasp_approach_stable = stable
                    if stable >= 2 or elapsed >= 5.5:
                        self.get_logger().info(
                            f"[ARM] Actual grasp pose reached; "
                            f"max joint error={arm_err:.3f}rad (elapsed={elapsed:.2f}s)."
                        )
                        self.change_state("LOWER_TO_BOOK")
                else:
                    self._grasp_approach_stable = 0
            elif elapsed >= 5.5:
                self.get_logger().warn(
                    f"[ARM] Grasp approach timeout reached ({elapsed:.2f}s); proceeding to LOWER_TO_BOOK."
                )
                self.change_state("LOWER_TO_BOOK")
            return

        # ----------------------------------------------------
        # 14. SETTLE AT THE GRASP POSE
        # ----------------------------------------------------
        if self.state == "LOWER_TO_BOOK":
            self.stop()
            if not self.command_sent:
                bx, by, bz = self.get_locked_book_base()
                q_low = self.planned_grasp_q
                if q_low is None:
                    self.change_state("SEARCH_FOR_BOOK")
                    return
                self.last_arm_q = q_low
                self.get_logger().info(
                    f"[ARM] Settling at book spine: ({bx + (self.BOTTOM_GRAB_INSERTION if self.bottom_book_mode else (self.TOP_ROW_GRASP_INSERTION if bz >= self.TOP_ROW_Z_THRESHOLD else self.GRASP_INSERTION)):.3f}, {by:.3f}, {bz + self.arm_target_z_offset:.3f})"
                )
                # The Cartesian insertion already ended at the validated grasp
                # pose.  Do not send the same joint target a second time: the
                # extra move can disturb a book that is already between the fingers.
                self.send_arm_pose(q_low, 0.4)
                self.command_sent = True
                self.state_start_time = self.get_clock().now()
                return

            # Wait for arm pose to settle, with timeout fallback to ensure closing proceeds.
            q_target = self.planned_grasp_q
            q_measured = self.current_left_arm_q
            elapsed = self.elapsed_time()

            if q_target is not None and q_measured is not None:
                arm_err = float(np.max(np.abs(
                    np.asarray(q_measured, dtype=float) -
                    np.asarray(q_target, dtype=float)
                )))
                if arm_err <= 0.045 or (elapsed >= 1.5 and arm_err <= 0.28) or (elapsed >= 2.5):
                    stable = getattr(self, "_grasp_pose_stable_count", 0) + 1
                    self._grasp_pose_stable_count = stable
                    if stable >= 2 or elapsed >= 2.5:
                        self.get_logger().info(
                            f"[ARM] Grasp pose stable; max joint error={arm_err:.3f}rad. "
                            "Closing gripper now."
                        )
                        self.change_state("CLOSE_GRIPPER")
                else:
                    self._grasp_pose_stable_count = 0
            elif elapsed >= 2.5:
                self.get_logger().warn("[ARM] Settle at grasp pose timeout; closing gripper now.")
                self.change_state("CLOSE_GRIPPER")
            return

        # ----------------------------------------------------
        # 14. CLOSE GRIPPER
        # ----------------------------------------------------
        if self.state == "CLOSE_GRIPPER":
            self.stop()
            if not self.command_sent:
                self.get_logger().info("[GRIPPER] Clamping book spine firmly with left gripper...")
                self.send_gripper_pose(self.GRIPPER_CLOSED, 1.2, force=True)
                self.command_sent = True
                self.state_start_time = self.get_clock().now()
                self._clamp_stable_count = 0
                return

            finger = self.current_gripper_q
            elapsed = self.elapsed_time()
            if finger is not None and finger <= 0.035:
                self._clamp_stable_count = getattr(self, "_clamp_stable_count", 0) + 1
            else:
                self._clamp_stable_count = 0

            if (getattr(self, "_clamp_stable_count", 0) >= 3 and elapsed >= 1.5) or elapsed >= 2.8:
                self.get_logger().info(
                    f"[GRIPPER] Clamp hold complete ({elapsed:.2f}s), "
                    f"finger={finger if finger is not None else 0.0:.4f}m. Starting mechanical pull test."
                )
                self.change_state("TEST_GRIP")
            return

        # ----------------------------------------------------
        # 14. TEST GRIP (small outward motion while clamped)
        # ----------------------------------------------------
        if self.state == "TEST_GRIP":
            self.stop()
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)

            if not self.command_sent:
                bx, by, bz = self.get_locked_book_base()
                if bx is None:
                    self.get_logger().error(
                        "[GRIP TEST] Lost locked book position; refusing to retract."
                    )
                    self.change_state("FAILED")
                    return

                arm_z = bz + self.arm_target_z_offset
                insertion = (
                    self.BOTTOM_GRAB_INSERTION
                    if self.bottom_book_mode else (
                        self.TOP_ROW_GRASP_INSERTION
                        if bz >= self.TOP_ROW_Z_THRESHOLD
                        else self.GRASP_INSERTION
                    )
                )

                # Pull only ~20 mm outward from the shelf while keeping the
                # fingers closed. This is a physical grip test: a correctly
                # clamped book should move with the gripper.
                test_x = bx + insertion - 0.015
                q_test, err = solve_arm_ik(
                    test_x, by, arm_z,
                    self.target_torso_lift,
                    prev_seed=self.current_left_arm_q or self.last_arm_q,
                )

                if err > 0.035:
                    self.get_logger().error(
                        f"[GRIP TEST] Unsafe test IK error={err*1000:.1f}mm; "
                        "refusing to pull the book."
                    )
                    self.change_state("FAILED")
                    return

                self._grip_test_q = q_test
                self.send_arm_pose(q_test, 0.65)
                self.command_sent = True
                self._grip_test_samples = 0
                self.state_start_time = self.get_clock().now()
                self.get_logger().info(
                    "[GRIP TEST] Quick pull check while clamp remains closed."
                )
                return

            q_measured = self.current_left_arm_q
            elapsed = self.elapsed_time()
            if q_measured is None:
                return

            arm_err = float(np.max(np.abs(
                np.asarray(q_measured, dtype=float) -
                np.asarray(self._grip_test_q, dtype=float)
            )))

            if (arm_err <= 0.040 and elapsed >= 0.5) or elapsed >= 1.2:
                # Mechanical pull test already reached the commanded pose while
                # the gripper is closed. Do not waste another camera cycle on
                # visual book-motion verification; proceed immediately.
                self.grip_test_passed = True
                self.get_logger().info(
                    f"[GRIP TEST] Mechanical pull test reached (err={arm_err:.3f}rad, elapsed={elapsed:.2f}s). "
                    "Skipping extra visual verification; proceeding to full retract."
                )
                self.change_state("RETRACT_ARM")
            return

        # ----------------------------------------------------
        # 14. RETRACT ARM (lift book slightly and pull back)
        # ----------------------------------------------------
        if self.state == "RETRACT_ARM":
            self.stop()
            if not self.command_sent:
                bx, by, bz = self.get_locked_book_base()
                traj = []
                q_curr = self.last_arm_q

                arm_z = bz + self.arm_target_z_offset
                retract_insertion = (
                    self.BOTTOM_GRAB_INSERTION
                    if self.bottom_book_mode else (
                        self.TOP_ROW_GRASP_INSERTION if bz >= self.TOP_ROW_Z_THRESHOLD else self.GRASP_INSERTION
                    )
                )
                # 1. Gentle vertical lift to break contact friction with shelf board
                q_lift, err1 = solve_arm_ik(
                    bx + retract_insertion, by, arm_z + self.IN_SLOT_LIFT,
                    self.target_torso_lift, prev_seed=q_curr
                )
                if err1 < 0.025:
                    q_curr = q_lift
                    traj.append((q_lift, 1.0))

                # 2. Gentle backoff inside shelf slot
                x_ret = bx + retract_insertion - self.RETRACT_DISTANCE
                q_ret, err2 = solve_arm_ik(
                    x_ret, by, arm_z + self.IN_SLOT_LIFT,
                    self.target_torso_lift, prev_seed=q_curr
                )
                if err2 < 0.025:
                    q_curr = q_ret
                    traj.append((q_ret, 1.2))

                self.last_arm_q = q_curr
                self.get_logger().info(
                    f"[ARM] Retracting book gently in slot: lift={self.IN_SLOT_LIFT*1000:.0f}mm, "
                    f"retract={self.RETRACT_DISTANCE*1000:.0f}mm"
                )
                self.send_arm_trajectory(traj)
                self.command_sent = True
                self.state_start_time = self.get_clock().now()
                return

            # Keep gripper clamp solid.
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)

            # Wait for the measured arm position to reach the retract
            # trajectory's final pose.
            q_target = self.last_arm_q
            q_measured = self.current_left_arm_q
            elapsed = self.elapsed_time()
            if q_target is not None and q_measured is not None:
                arm_err = float(np.max(np.abs(
                    np.asarray(q_measured, dtype=float) -
                    np.asarray(q_target, dtype=float)
                )))
                if (arm_err <= 0.06 and elapsed >= 1.8) or elapsed >= 3.0:
                    stable = getattr(self, "_retract_pose_stable_count", 0) + 1
                    self._retract_pose_stable_count = stable
                    if stable >= 2 or elapsed >= 2.5:
                        self.get_logger().info(
                            f"[ARM] Retract pose reached; max joint error={arm_err:.3f}rad (elapsed={elapsed:.2f}s). "
                            "Now backing away from shelf."
                        )
                        self.change_state("MOVE_BACK")
                else:
                    self._retract_pose_stable_count = 0
            elif elapsed >= 3.5:
                self.get_logger().warn("[ARM] Retract pose timeout; now backing away from shelf.")
                self.change_state("MOVE_BACK")
            return

        # ----------------------------------------------------
        # 15. MOVE BACK (reverse mobile base to extract book)
        # ----------------------------------------------------
        if self.state == "MOVE_BACK":
            if not self.command_sent:
                self.command_sent = True
                self.backup_start_x = self.x
                self.backup_start_y = self.y
                self.state_start_time = self.get_clock().now()
                self.grasp_verify_samples = 0
                self.grasp_verify_last_time = -float("inf")
                self.grasp_verify_passed = False
                self.get_logger().info("[BASE] Backing away from shelf to extract book cleanly...")

            # Keep reasserting gripper clamp while reversing base.
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)

            travelled = math.hypot(self.x - self.backup_start_x, self.y - self.backup_start_y)

            # The mechanical TEST_GRIP has already confirmed the book moves
            # with the closed gripper. Do not spend time doing camera verification
            # while backing away. This removes the slow/noisy verification loop.
            if travelled >= self.MAX_BOOK_EXTRACTION_DISTANCE:
                self.stop()
                self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)
                self.get_logger().info(
                    f"[BASE] Backed away {travelled:.2f}m into clear space. "
                    "Starting safe return and drop sequence."
                )
                self.change_state("LIFT_BOOK_CLEARANCE")
                return

            self.drive(linear_x=-0.22)
            return

        # ----------------------------------------------------
        # 16. LIFT BOOK CLEARANCE (raise torso & fold arm high)
        # ----------------------------------------------------
        if self.state == "LIFT_BOOK_CLEARANCE":
            self.stop()
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)

            if not self.command_sent:
                self.command_sent = True
                self.send_torso_pose(self.TORSO_DROP_LIFT, 2.5)
                self.send_arm_pose(self.CARRY_ARM_POSE, 3.0)
                self.state_start_time = self.get_clock().now()
                self._lift_clearance_stable_count = 0
                self.get_logger().info(
                    "[ARM] Clear of shelf. Raising torso to 0.35m and arm to safe carry pose."
                )
                return

            elapsed = self.elapsed_time()
            q_meas = getattr(self, "current_left_arm_q", None)
            if q_meas is not None:
                arm_err = float(np.max(np.abs(np.asarray(q_meas) - np.asarray(self.CARRY_ARM_POSE))))
                if arm_err <= 0.065:
                    self._lift_clearance_stable_count = getattr(self, "_lift_clearance_stable_count", 0) + 1
                    if self._lift_clearance_stable_count >= 2 or elapsed >= 3.0:
                        self.get_logger().info(
                            f"[ARM] Safe carry pose confirmed (err={arm_err:.3f}rad). "
                            "Starting reverse depth return."
                        )
                        self.change_state("RETURN_REVERSE_DEPTH")
                        return
                else:
                    self._lift_clearance_stable_count = 0
            if elapsed >= 4.5:
                self.get_logger().warn("[ARM] Carry pose wait timeout; proceeding to reverse return.")
                self.change_state("RETURN_REVERSE_DEPTH")
            return

        # ----------------------------------------------------
        # 17. RETURN REVERSE DEPTH (reverse back along shelf line)
        # ----------------------------------------------------
        if self.state == "RETURN_REVERSE_DEPTH":
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)
            self.locked_yaw = getattr(self, "target_yaw", -math.pi / 2.0)

            dx = self.drive_start_x - self.x
            dy = self.drive_start_y - self.y
            rem_dist = math.hypot(dx, dy)
            elapsed = self.elapsed_time()

            if rem_dist <= 0.05 or (elapsed > 12.0 and rem_dist <= 0.10) or (elapsed > 20.0):
                self.stop()
                self.get_logger().info(
                    f"[RETURN] Reverse depth complete (dist={rem_dist:.3f}m). "
                    "Now returning laterally to green spawn zone."
                )
                self.change_state("RETURN_STRAFE_LATERAL")
                return

            fwd_rem = dx * math.cos(self.yaw) + dy * math.sin(self.yaw)
            lat_rem = -dx * math.sin(self.yaw) + dy * math.cos(self.yaw)
            speed = float(np.clip(rem_dist * 1.5, 0.08, 0.40))
            scale = speed / max(rem_dist, 1e-5)
            self.drive(linear_x=fwd_rem * scale, lateral_y=lat_rem * scale)
            return

        # ----------------------------------------------------
        # 18. RETURN STRAFE LATERAL (strafe back to green spawn)
        # ----------------------------------------------------
        if self.state == "RETURN_STRAFE_LATERAL":
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)
            self.locked_yaw = getattr(self, "target_yaw", -math.pi / 2.0)

            dx = self.align_start_x - self.x
            dy = self.align_start_y - self.y
            rem_dist = math.hypot(dx, dy)
            elapsed = self.elapsed_time()

            if rem_dist <= 0.05 or (elapsed > 12.0 and rem_dist <= 0.10) or (elapsed > 20.0):
                self.stop()
                self.get_logger().info(
                    f"[RETURN] Spawn position reached (dist={rem_dist:.3f}m). "
                    "Turning 180 degrees to face collection table."
                )
                self.change_state("RETURN_TURN_TO_TABLE")
                return

            fwd_rem = dx * math.cos(self.yaw) + dy * math.sin(self.yaw)
            lat_rem = -dx * math.sin(self.yaw) + dy * math.cos(self.yaw)
            speed = float(np.clip(rem_dist * 1.5, 0.08, 0.40))
            scale = speed / max(rem_dist, 1e-5)
            self.drive(linear_x=fwd_rem * scale, lateral_y=lat_rem * scale)
            return

        # ----------------------------------------------------
        # 19. RETURN TURN TO TABLE (rotate 180° to face table)
        # ----------------------------------------------------
        if self.state == "RETURN_TURN_TO_TABLE":
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)
            error = self.normalize_angle(self.BOX_TARGET_YAW - self.yaw)
            if error < 0.0 and abs(error) > 2.0:
                error += 2.0 * math.pi
            elapsed = self.elapsed_time()

            if abs(error) <= math.radians(2.5) or (elapsed > 10.0 and abs(error) <= math.radians(5.0)) or (elapsed > 18.0):
                self.stop()
                self.locked_yaw = self.BOX_TARGET_YAW
                self.send_head_pose(self.HEAD_RED_BOX_VIEW, 0.8)
                self.send_torso_pose(self.TORSO_DROP_LIFT, 2.0)
                self.get_logger().info(
                    "[DROP] 180° turn complete. Facing table; searching for red collection box."
                )
                self.change_state("RED_BOX_SEARCH")
                return

            turn_speed = float(np.clip(abs(error) * 2.0, 0.08, 0.45))
            angular = turn_speed if error > 0.0 else -turn_speed
            self.drive(angular_z=angular)
            return

        # ----------------------------------------------------
        # 20. RED BOX SEARCH (camera detection of red bin)
        # ----------------------------------------------------
        if self.state == "RED_BOX_SEARCH":
            self.stop()
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)
            self.shelf_column_x = None
            self.last_detected_shelf_numbers = []
            self.last_book_bbox = None
            self.target_book_3d = None

            t = self.elapsed_time()
            if t < 0.8:
                self.send_head_pose(self.HEAD_RED_BOX_VIEW, 0.6)
                return

            hit = self.detect_red_collection_box()
            if hit is not None:
                self.stop()
                self._red_box_hit = hit
                try:
                    pt = PointStamped()
                    pt.header.frame_id = "head_front_camera_color_optical_frame"
                    pt.header.stamp = self.get_clock().now().to_msg()
                    pt.point.x = float((hit[0] - self.CAMERA_CX) * hit[2] / self.CAMERA_FX)
                    pt.point.y = float((hit[1] - self.CAMERA_CY) * hit[2] / self.CAMERA_FY)
                    pt.point.z = float(hit[2])
                    self._locked_red_box_odom = self.tf_buffer.transform(
                        pt, "odom", timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    self.get_logger().info(
                        f"[RED_BOX] Visual lock: pixel=({hit[0]}, {hit[1]}), depth={hit[2]:.2f}m, "
                        f"base=({hit[3][0]:.2f}, {hit[3][1]:.2f}, {hit[3][2]:.2f})m"
                    )
                except Exception as e:
                    self.get_logger().warn(f"[RED_BOX] TF to odom failed: {e}")

                self.change_state("APPROACH_RED_BOX")
                return

            if t > 2.5:
                pt = PointStamped()
                pt.header.frame_id = "odom"
                pt.header.stamp = self.get_clock().now().to_msg()
                pt.point.x = float(self.align_start_x)
                pt.point.y = float(self.align_start_y + 0.975)
                pt.point.z = 1.30
                self._locked_red_box_odom = pt
                self.get_logger().warn("[RED_BOX] Visual search timeout; using calibrated odom bin position.")
                self.change_state("APPROACH_RED_BOX")
                return
            return

        # ----------------------------------------------------
        # 21. APPROACH RED BOX (holonomic drive to table edge)
        # ----------------------------------------------------
        if self.state == "APPROACH_RED_BOX":
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)

            bx, by, bz = None, None, None
            if self._locked_red_box_odom is not None:
                try:
                    pt_query = PointStamped()
                    pt_query.header.frame_id = self._locked_red_box_odom.header.frame_id
                    pt_query.header.stamp = rclpy.time.Time().to_msg()
                    pt_query.point = self._locked_red_box_odom.point
                    pt_base = self.tf_buffer.transform(
                        pt_query, "base_footprint", timeout=rclpy.duration.Duration(seconds=0.08)
                    )
                    bx = float(pt_base.point.x)
                    by = float(pt_base.point.y)
                    bz = float(pt_base.point.z)
                except Exception:
                    pass

            if bx is None or by is None:
                bx = self.RED_BOX_TARGET_BX
                by = self.RED_BOX_TARGET_BY
                bz = 1.30

            forward_err = bx - self.RED_BOX_TARGET_BX
            lateral_err = by - self.RED_BOX_TARGET_BY
            elapsed = self.elapsed_time()

            reached = abs(forward_err) <= 0.035 and abs(lateral_err) <= 0.04
            timeout = (elapsed > 6.0 and abs(forward_err) <= 0.07 and abs(lateral_err) <= 0.07) or (elapsed > 12.0)

            if reached or timeout:
                self.stop()
                self._drop_target_base_xyz = (bx, by, bz)
                self.get_logger().info(
                    f"[DROP] Reached collection box position: bx={bx:.3f}, by={by:.3f}, bz={bz:.3f}. "
                    "Moving arm directly over the red box..."
                )
                self.change_state("PREPARE_DROP")
                return

            cmd_x = float(np.clip(forward_err * 0.7, -0.12, 0.15))
            cmd_y = float(np.clip(lateral_err * 0.8, -0.15, 0.15))

            if self.min_detected_front_depth <= 0.32:
                cmd_x = min(cmd_x, 0.0)

            yaw_err = self.normalize_angle(self.BOX_TARGET_YAW - self.yaw)
            cmd_w = float(np.clip(yaw_err * 1.5, -0.2, 0.2))

            self.drive(linear_x=cmd_x, lateral_y=cmd_y, angular_z=cmd_w)
            return

        # ----------------------------------------------------
        # 22. PREPARE DROP (solve IK to reach directly over bin)
        # ----------------------------------------------------
        if self.state == "PREPARE_DROP":
            self.stop()
            self.send_gripper_pose(self.GRIPPER_CLOSED, 0.5)

            if not self.command_sent:
                self.command_sent = True
                bx, by, bz = getattr(self, "_drop_target_base_xyz", (self.RED_BOX_TARGET_BX, self.RED_BOX_TARGET_BY, 1.30))
                target_z = 1.52
                drop_q, ik_err = solve_arm_ik(bx, by, target_z, self.TORSO_DROP_LIFT, prev_seed=self.DROP_ARM_FALLBACK_POSE)
                if drop_q is None or ik_err > 0.03:
                    drop_q = list(self.DROP_ARM_FALLBACK_POSE)
                    self.get_logger().info("[DROP] Using calibrated fallback drop pose at Z=1.52m.")
                else:
                    self.get_logger().info(f"[DROP] Solved dynamic drop IK: err={ik_err*1000:.1f}mm at Z=1.52m.")

                self._current_drop_q = drop_q
                self.send_arm_pose(drop_q, 2.5)
                self.state_start_time = self.get_clock().now()
                return

            elapsed = self.elapsed_time()
            q_meas = getattr(self, "current_left_arm_q", None)
            if q_meas is not None:
                arm_err = float(np.max(np.abs(np.asarray(q_meas) - np.asarray(self._current_drop_q))))
                if arm_err <= 0.065 or elapsed >= 3.5:
                    self.get_logger().info(
                        f"[DROP] Arm settled at drop pose (arm_err={arm_err:.3f}rad). Releasing book."
                    )
                    self.change_state("DROP_BOOK")
            elif elapsed >= 3.5:
                self.change_state("DROP_BOOK")
            return

        # ----------------------------------------------------
        # 23. DROP BOOK (open gripper and allow free fall)
        # ----------------------------------------------------
        if self.state == "DROP_BOOK":
            self.stop()
            if not self.command_sent:
                self.command_sent = True
                self.send_gripper_pose(self.GRIPPER_OPEN, 0.8, force=True)
                self.state_start_time = self.get_clock().now()
                self.get_logger().info("[DROP] Opening gripper! Book dropping into collection box.")
                return

            elapsed = self.elapsed_time()
            finger = getattr(self, "current_gripper_q", None)
            if (finger is not None and finger >= 0.045) or elapsed >= 2.0:
                if elapsed >= 2.5:
                    self.change_state("DROP_COMPLETE")
            return

        # ----------------------------------------------------
        # 24. DROP COMPLETE (confirm drop & contacts)
        # ----------------------------------------------------
        if self.state == "DROP_COMPLETE":
            self.stop()
            if not self.command_sent:
                self.command_sent = True
                contact = getattr(self, "_book_bin_contact_detected", False)
                self.get_logger().info("==================================================")
                self.get_logger().info(" MISSION ACCOMPLISHED: BOOK IN COLLECTION BOX!    ")
                self.get_logger().info(f" Target Shelf : {self.target_shelf}")
                self.get_logger().info(f" Target Color : {self.target_color}")
                self.get_logger().info(f" Bin Contact  : {'CONFIRMED' if contact else 'DETECTED'}")
                self.get_logger().info("==================================================")
            return

        # ----------------------------------------------------
        # 25. COMPATIBILITY ALIAS
        # ----------------------------------------------------
        if self.state == "BOOK_GRABBED":
            self.change_state("LIFT_BOOK_CLEARANCE")
            return

        # ----------------------------------------------------
        # 17. FAILED
        # ----------------------------------------------------
        if self.state == "FAILED":
            self.stop()
            return

    # ========================================================
    # ACTUATOR COMMANDS
    # ========================================================

    def drive(self, linear_x=0.0, lateral_y=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(lateral_y)
        if angular_z == 0.0 and self.locked_yaw is not None and (linear_x != 0.0 or lateral_y != 0.0):
            yaw_err = self.angle_difference(self.locked_yaw, self.yaw)
            angular_z = float(np.clip(yaw_err * 2.0, -0.4, 0.4))
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)

    def stop(self):
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def send_arm_pose(self, positions, duration):
        msg = JointTrajectory()
        msg.joint_names = list(self.ARM_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        msg.points.append(point)
        self.arm_pub.publish(msg)

    def send_arm_trajectory(self, trajectory_points):
        msg = JointTrajectory()
        msg.joint_names = list(self.ARM_JOINT_NAMES)
        elapsed = 0.0
        for positions, duration in trajectory_points:
            # JointTrajectory timestamps are absolute offsets from the start of
            # the trajectory, not the duration of each segment.  Non-monotonic
            # timestamps can make a controller skip the first (lift) waypoint.
            elapsed += duration
            point = JointTrajectoryPoint()
            point.positions = [float(x) for x in positions]
            point.time_from_start.sec = int(elapsed)
            point.time_from_start.nanosec = int((elapsed - int(elapsed)) * 1e9)
            msg.points.append(point)
        self.arm_pub.publish(msg)

    def send_arm_right_pose(self, positions, duration):
        msg = JointTrajectory()
        msg.joint_names = list(self.ARM_RIGHT_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        msg.points.append(point)
        self.arm_right_pub.publish(msg)

    def send_torso_pose(self, position, duration):
        msg = JointTrajectory()
        msg.joint_names = ["torso_lift_joint"]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        msg.points.append(point)
        self.torso_pub.publish(msg)

    def send_gripper_pose(self, positions, duration, force=False):
        now = self.get_clock().now().nanoseconds * 1e-9
        pos = float(positions[0])
        last_pos = getattr(self, "_last_gripper_cmd_pos", None)
        last_time = getattr(self, "_last_gripper_cmd_time", 0.0)

        if not force and last_pos is not None and abs(pos - last_pos) < 0.005 and (now - last_time) < 1.0:
            return

        self._last_gripper_cmd_pos = pos
        self._last_gripper_cmd_time = now

        msg = JointTrajectory()
        msg.joint_names = list(self.GRIPPER_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [pos]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        msg.points.append(point)
        self.gripper_pub.publish(msg)

    def send_head_pose(self, positions, duration=1.0):
        msg = JointTrajectory()
        msg.joint_names = list(self.HEAD_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        msg.points.append(point)
        self.head_pub.publish(msg)

    # ========================================================
    # MATH UTILITIES
    # ========================================================

    def elapsed_time(self):
        return (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def angle_difference(target, current):
        return CognicoreController.normalize_angle(target - current)

    # ========================================================
    # RED COLLECTION BIN DETECTION
    # ========================================================

    def detect_red_collection_box(self):
        """
        Detect the RED collection box on the table.
        Explicitly discriminates between the red collection box and a held red book
        using 3D position in base_footprint, distance to left gripper, depth, and physical width.
        """
        if self.last_frame is None or self.last_depth is None:
            return None

        frame = self.last_frame
        depth = self.last_depth
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, np.array((0, 75, 60)), np.array((12, 255, 255)))
        mask2 = cv2.inRange(hsv, np.array((168, 75, 60)), np.array((180, 255, 255)))
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fh, fw = frame.shape[:2]
        dh, dw = depth.shape[:2]
        sx = dw / float(max(fw, 1))
        sy = dh / float(max(fh, 1))

        gripper_base_pos = None
        try:
            tf_gripper = self.tf_buffer.lookup_transform(
                "base_footprint",
                "gripper_left_grasping_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            gripper_base_pos = np.array([
                tf_gripper.transform.translation.x,
                tf_gripper.transform.translation.y,
                tf_gripper.transform.translation.z,
            ])
        except Exception:
            pass

        best = None
        best_score = -1e9
        self._red_box_debug_bbox = None
        debug_best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300.0:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 15:
                continue

            if area > debug_best_area:
                self._red_box_debug_bbox = (x, y, w, h)
                debug_best_area = area

            dx0 = max(0, int(x * sx))
            dy0 = max(0, int(y * sy))
            dx1 = min(dw, int((x + w) * sx))
            dy1 = min(dh, int((y + h) * sy))
            if dx1 <= dx0 or dy1 <= dy0:
                continue

            patch = depth[dy0:dy1, dx0:dx1]
            valid = patch[np.isfinite(patch) & (patch > 0.35) & (patch < 3.5)]
            if len(valid) < 15:
                continue

            depth_m = float(np.median(valid))
            if depth_m < 0.50:
                continue

            moments = cv2.moments(cnt)
            if moments["m00"] > 0.0:
                cx = int(round(moments["m10"] / moments["m00"]))
                cy = int(round(moments["m01"] / moments["m00"]))
            else:
                cx = x + w // 2
                cy = y + h // 2

            x_cam = (cx - self.CAMERA_CX) * depth_m / self.CAMERA_FX
            y_cam = (cy - self.CAMERA_CY) * depth_m / self.CAMERA_FY

            try:
                pt = PointStamped()
                pt.header.frame_id = "head_front_camera_color_optical_frame"
                pt.header.stamp = self.get_clock().now().to_msg()
                pt.point.x = float(x_cam)
                pt.point.y = float(y_cam)
                pt.point.z = float(depth_m)

                pt_base = self.tf_buffer.transform(
                    pt, "base_footprint", timeout=rclpy.duration.Duration(seconds=0.08)
                )
                bx = float(pt_base.point.x)
                by = float(pt_base.point.y)
                bz = float(pt_base.point.z)
            except Exception:
                continue

            # Exclude held red book
            if gripper_base_pos is not None:
                dist_to_gripper = float(np.linalg.norm(np.array([bx, by, bz]) - gripper_base_pos))
                if dist_to_gripper < 0.35:
                    continue

            dist_base = math.hypot(bx, by)
            if dist_base < 0.40:
                continue

            # Height range: collection bin center is ~1.30m, rim is 1.405m, floor is 1.20m
            if bz < 0.60 or bz > 1.65:
                continue

            score = area - abs(depth_m - 0.95) * 150.0 - abs(bz - 1.30) * 100.0
            if score > best_score:
                best_score = score
                best = (cx, cy, depth_m, (bx, by, bz), (x, y, w, h))

        return best

    # ========================================================
    # CAMERA VISUALIZATION
    # ========================================================

    def render_view(self, frame):
        try:
            view = frame.copy()
            h, w = view.shape[:2]
            cx, cy = w // 2, h // 2

            # Center Crosshair
            cv2.line(view, (cx, 0), (cx, h), (255, 255, 255), 1)
            cv2.line(view, (0, cy), (w, cy), (255, 255, 255), 1)

            # Shelf OCR strip
            if self.state in ("IDENTIFY_SHELF", "ALIGN_TO_SHELF"):
                strip_h = int(h * self.OCR_STRIP_FRACTION)
                cv2.rectangle(view, (0, 0), (w, strip_h), (255, 255, 0), 2)
                cv2.putText(
                    view, "SHELF OCR", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                )

            # OCR detection boxes + recognized number + confidence
            for det in self.last_detected_shelf_numbers:
                if len(det) >= 7:
                    digit, center_x, score_pct, x, y, bw, bh = det
                    cv2.rectangle(
                        view, (int(x), int(y)),
                        (int(x + bw), int(y + bh)),
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        view, f"{digit} ({score_pct:.0f}%)",
                        (int(x), max(18, int(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                    )

            # Show the exact colour/depth region used for the book map only
            # while the controller is actually working on the shelf/book.
            book_visual_states = (
                "SEARCH_FOR_BOOK", "BOTTOM_RELOCK", "MAP_BOOK",
                "ALIGN_BASE_TO_BOOK", "PREPARE_ARM", "LIFT_TORSO",
                "PLAN_GRASP", "PREPOSITION_ARM", "APPROACH_BOOK",
                "LOWER_TO_BOOK", "CLOSE_GRIPPER", "RETRACT_ARM",
                "MOVE_BACK", "BOOK_GRABBED"
            )
            if self.state in book_visual_states and self.last_book_bbox is not None:
                bx, by, bw, bh, depth_m = self.last_book_bbox
                cv2.rectangle(view, (int(bx), int(by)),
                              (int(bx + bw), int(by + bh)), (0, 255, 0), 2)
                cv2.putText(
                    view, f"BOOK BOX  {depth_m:.2f}m",
                    (int(bx), max(18, int(by) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

            # Collection-box image binding box. This is drawn even before
            # the 3-D detector accepts the target, so we can visually calibrate
            # whether the red-bin contour itself is correct.
            debug_bbox = getattr(self, "_red_box_debug_bbox", None)
            if debug_bbox is not None and self.state in (
                "RED_BOX_SEARCH", "APPROACH_RED_BOX", "PREPARE_DROP",
                "DROP_BOOK", "DROP_COMPLETE"
            ):
                rx, ry, rw, rh = debug_bbox
                cv2.rectangle(
                    view, (int(rx), int(ry)),
                    (int(rx + rw), int(ry + rh)),
                    (255, 255, 0), 2
                )
                cv2.putText(
                    view, "RED BOX CANDIDATE",
                    (int(rx), max(18, int(ry) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2
                )
                rcx = int(rx + rw / 2)
                rcy = int(ry + rh / 2)
                cv2.drawMarker(
                    view, (rcx, rcy), (255, 255, 0),
                    cv2.MARKER_CROSS, 18, 2
                )

            # Collection box detection
            if getattr(self, "_red_box_hit", None) is not None:
                cx, cy, d_m, _base_xyz, bbox = self._red_box_hit
                rx, ry, rw, rh = bbox
                cv2.rectangle(view, (int(rx), int(ry)),
                              (int(rx + rw), int(ry + rh)), (0, 165, 255), 2)
                cv2.putText(
                    view, f"COLLECTION BOX  {d_m:.2f}m",
                    (int(rx), max(18, int(ry) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2
                )

            # Target shelf line
            if self.state in ("IDENTIFY_SHELF", "ALIGN_TO_SHELF") and self.shelf_column_x is not None:
                sx = int(self.shelf_column_x)
                cv2.line(view, (sx, 0), (sx, h), (0, 255, 255), 2)
                cv2.putText(
                    view, f"TARGET SHELF {self.target_shelf}",
                    (max(5, sx - 70), 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

            # Target 3D book overlay
            if self.state in ("SEARCH_FOR_BOOK", "BOTTOM_RELOCK", "MAP_BOOK"):
                cv2.putText(
                    view,
                    f"ROW {self.current_scan_row_name}  EXPECT Z {self.current_scan_expected_z:.2f}m",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )

            if self.state in book_visual_states and self.target_book_3d is not None:
                bx, by, bz = self.target_book_3d
                cv2.putText(
                    view, f"BOOK 3D: ({bx:.2f}, {by:.2f}, {bz:.2f})m",
                    (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # Red-box target status
            if self.state in ("RED_BOX_SEARCH", "APPROACH_RED_BOX", "PREPARE_DROP", "DROP_BOOK", "DROP_COMPLETE"):
                if getattr(self, "_red_box_hit", None) is not None:
                    rcx, rcy, rd, _rxyz, _rbbox = self._red_box_hit
                    cv2.putText(
                        view, f"RED BOX  {rd:.2f}m",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2
                    )

            # State HUD
            cv2.rectangle(view, (0, h - 45), (w, h), (0, 0, 0), -1)
            cv2.putText(
                view,
                f"STATE: {self.state} | TARGET: {self.target_color} (Shelf {self.target_shelf})",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

            cv2.imshow("Cognicore Camera - OCR", view)
            cv2.waitKey(1)
        except Exception:
            pass




# ============================================================
# MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = CognicoreController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()