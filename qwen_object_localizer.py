#!/usr/bin/env python3
"""
qwen_object_localizer.py

YOLO-World 탐지 결과에 LiDAR 거리 또는 박스 크기 거리를 결합해 물체의 위치를 추정한다.
(ROS 2 노드와 계산 함수. find_object_demo.py가 계산 함수를 불러 쓴다.)

실행 예:
  python3 qwen_object_localizer.py --classes cup bottle --heights cup=10 bottle=16
  python3 qwen_object_localizer.py --qwen --heights cup=10   (한국어 명령 입력, qwen_command.py 필요)
"""

import argparse
import math
import threading
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data
)

from sensor_msgs.msg import Image, CompressedImage, LaserScan
from ultralytics import YOLO


# 카메라 내부 파라미터 (640x480 보정값). 사용하는 카메라에 맞게 수정한다.

FX = 641.1
FY = 644.4
CX = 311.6
CY = 257.3

CALIBRATION_WIDTH = 640
CALIBRATION_HEIGHT = 480


# 카메라가 LiDAR보다 앞쪽에 있는 거리(m)
CAMERA_FORWARD_OFFSET_M = 0.06

# 카메라-LiDAR 방향 오차
CAMERA_LIDAR_ANGLE_OFFSET_DEG = 0.0

# YOLO 박스는 실제 물체보다 약 9% 크게 잡혀서, 실제 높이에 곱해 보정한다
BOX_MARGIN = 1.09

# 물체 실제 높이(cm) 기본값. --heights로 덮어쓴다.
DEFAULT_HEIGHTS_CM = {"book": 28.0, "bottle": 16.0}

# LiDAR 스캔 평면의 바닥 기준 높이(cm). 이보다 낮은 물체는 LiDAR가 볼 수 없다.
LIDAR_PLANE_HEIGHT_CM = 19.0


def nv21_to_bgr(msg: Image) -> np.ndarray:

    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    yuv = buf.reshape((msg.height * 3 // 2, msg.step))[:, :msg.width].copy()

    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)


def compressed_to_bgr(msg: CompressedImage) -> np.ndarray:

    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ROS 없이 테스트할 수 있는 계산 함수

def parse_heights(items):
    heights = dict(DEFAULT_HEIGHTS_CM)
    for it in items or []:
        name, value = it.split("=")
        heights[name.strip()] = float(value)
    return heights


def lidar_lookup(scan, angle_deg, window):
    # 카메라 각도(deg, 왼쪽 +)에 해당하는 LiDAR 거리(m).
    # 해당 빔과 좌우 window개 빔 중 유효한 값의 최솟값을 사용한다.
    # 반환: (각도[rad], 거리[m] 또는 None)

    target = math.radians(angle_deg + CAMERA_LIDAR_ANGLE_OFFSET_DEG)

    target = scan.angle_min + ((target - scan.angle_min) % (2.0 * math.pi))

    n = len(scan.ranges)

    center = round((target - scan.angle_min) / scan.angle_increment)

    valid = []

    for k in range(-window, window + 1):

        idx = (center + k) % n

        r = scan.ranges[idx]

        if (
            math.isfinite(r)
            and scan.range_min <= r <= scan.range_max
        ):
            valid.append(r)

    if not valid:
        return target, None

    return target, min(valid)


def box_estimate(u, y1, y2, img_h, height_cm):
    # 박스 크기로 거리를 추정한다.
    # 반환: dict(z_cam[m], x[m], y[m]) 또는 None. x, y는 로봇 좌표(x 앞, y 왼쪽 +).

    if not height_cm:
        return None

    h_px = y2 - y1

    if h_px <= 0:
        return None

    # 박스가 이미지 위/아래 끝에 걸려 잘리면 제외한다.
    if y1 <= 12 or y2 >= img_h - 12:
        return None

    z_cam = FY * (height_cm * BOX_MARGIN) / h_px / 100.0

    x_right = (u - CX) / FX * z_cam

    return {
        "z_cam": z_cam,
        "x": z_cam + CAMERA_FORWARD_OFFSET_M,
        "y": -x_right,
    }


def height_estimate(z_cam, y1, y2, cam_height_m=0.135, pitch_deg=0.0):
    # 박스의 위/중심/아래 끝이 바닥에서 몇 m 높이인지 계산한다.
    # 위/아래 끝의 거리가 같다고 가정하므로, 물체가 바로 서 있을 때만 의미가 있다.
    # pitch_deg: 카메라가 아래로 기울어진 각도. 수평이면 0.

    th = math.radians(pitch_deg)

    def z_at(v):
        yc = (v - CY) * z_cam / FY
        return cam_height_m - z_cam * math.sin(th) - yc * math.cos(th)

    return {
        "top": z_at(y1),
        "center": z_at((y1 + y2) / 2.0),
        "bottom": z_at(y2),
    }


def choose_estimate(lidar, box, mismatch_ratio, lidar_blocked=False):
    # LiDAR 우선. 물체가 LiDAR 평면보다 낮거나(lidar_blocked),
    # LiDAR가 박스 추정보다 mismatch_ratio배 이상 멀면 박스 거리를 사용한다.
    # 반환: (출처 'lidar'/'box'/'none', 선택된 값 또는 None, 메모)

    if lidar_blocked:

        if box is not None:

            return (
                "box",
                box,
                "물체 높이가 LiDAR 평면보다 낮아 LiDAR 값은 사용하지 않음"
            )

        return "none", None, "물체가 LiDAR 평면보다 낮고 박스 추정도 불가"

    if lidar is not None:

        if box is not None and lidar["z_cam"] > box["z_cam"] * mismatch_ratio:

            ratio = lidar["z_cam"] / box["z_cam"]

            return (
                "box",
                box,
                f"LiDAR가 박스 추정보다 {ratio:.2f}배 멀어 "
                f"스캔 평면 아래 물체로 판단"
            )

        return "lidar", lidar, ""

    if box is not None:
        return "box", box, "LiDAR 거리 없음"

    return "none", None, "LiDAR 거리 없음, 박스 추정 불가"


# ROS 2 노드

class ObjectLocalizerNode(Node):

    def __init__(
        self,
        target_classes,
        conf_threshold,
        infer_hz,
        imgsz,
        use_compressed,
        heights_cm,
        mismatch_ratio,
        beam_window,
        lidar_height_cm=LIDAR_PLANE_HEIGHT_CM,
        image_topic=None
    ):

        super().__init__("qwen_object_localizer")

        self.lidar_height_cm = lidar_height_cm

        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.use_compressed = use_compressed
        self.heights_cm = heights_cm
        self.mismatch_ratio = mismatch_ratio
        self.beam_window = beam_window

        self.get_logger().info(
            f"YOLO-World 로딩 중. 대상: {target_classes}"
        )

        self.model = YOLO("yolov8s-world.pt")
        self.model.set_classes(target_classes)

        # 명령으로 바뀐 탐지 대상은 바로 적용하지 않고 추론 타이머가 다음 실행 때 적용.
        self.pending_classes = None
        self.class_lock = threading.Lock()

        self.latest_scan = None
        self.latest_image_msg = None

        self.camera_received = False
        self.lidar_received = False


        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        if use_compressed:

            self.create_subscription(
                CompressedImage,
                image_topic or '/camera/image_raw/compressed',
                self.image_callback,
                image_qos
            )

        else:

            self.create_subscription(
                Image,
                image_topic or '/camera/image_raw',
                self.image_callback,
                image_qos
            )


        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )


        self.create_timer(
            1.0 / infer_hz,
            self.infer_callback
        )

        self.get_logger().info(
            f"준비 완료 — 추론 {infer_hz}Hz, "
            f"imgsz={imgsz}, "
            f"{'compressed' if use_compressed else 'raw'}"
        )

        self.get_logger().info(
            f"Camera Intrinsics: "
            f"FX={FX}, FY={FY}, CX={CX}, CY={CY}"
        )

        self.get_logger().info(
            f"Camera-LiDAR: 앞쪽 오프셋 {CAMERA_FORWARD_OFFSET_M * 100:.0f}cm, "
            f"각도 오프셋 {CAMERA_LIDAR_ANGLE_OFFSET_DEG}deg, "
            f"빔 윈도우 ±{beam_window}"
        )

        self.get_logger().info(
            f"박스 보조 경로 높이(cm): {heights_cm}, "
            f"여유 x{BOX_MARGIN}, 불일치 기준 x{mismatch_ratio}, "
            f"LiDAR 평면 높이 {self.lidar_height_cm}cm"
        )


    def request_class(self, cls):
        with self.class_lock:
            self.pending_classes = [cls]

    def command_loop(self, use_qwen_model, qwen_model):
        from qwen_command import get_object_name

        print("한국어 명령을 입력하세요 (예: 컵 찾아줘). 종료는 Ctrl+C")

        while rclpy.ok():
            try:
                text = input("명령> ")
            except EOFError:
                return

            cls, src, note = get_object_name(
                text, use_qwen=use_qwen_model, model=qwen_model
            )

            if cls is None:
                print(f"알아듣지 못함 ({note})")
                continue

            msg = f"  -> '{cls}' (출처: {src})"
            if cls not in self.heights_cm:
                msg += " | 높이 정보 없음: LiDAR 값만 사용 (--heights 로 추가 가능)"
            print(msg)

            self.request_class(cls)


    def scan_callback(self, msg: LaserScan):

        self.latest_scan = msg

        if not self.lidar_received:

            self.lidar_received = True

            self.get_logger().info(
                f"[SENSOR] LiDAR 수신 성공: /scan "
                f"(포인트 수: {len(msg.ranges)})"
            )


    def image_callback(self, msg):

        self.latest_image_msg = msg

        if not self.camera_received:

            self.camera_received = True

            # CompressedImage에는 width/height가 없어 해상도 대신 형식과 크기를 출력.
            size = (
                f"{msg.width}x{msg.height}"
                if hasattr(msg, "width")
                else f"압축({msg.format}, {len(msg.data)}바이트)"
            )

            self.get_logger().info(
                f"[SENSOR] Camera 수신 성공 (해상도: {size})"
            )


    def pixel_to_angle_deg(self, u):

        # 이미지 중심보다 왼쪽이면 양수, 오른쪽이면 음수인 각도
        return math.degrees(math.atan2(CX - u, FX))


    def infer_callback(self):

        with self.class_lock:
            new_classes = self.pending_classes
            self.pending_classes = None

        if new_classes is not None:
            self.model.set_classes(new_classes)
            self.target_classes = new_classes
            self.get_logger().info(f"탐지 대상 변경: {new_classes}")

        msg = self.latest_image_msg

        if msg is None:
            return

        # 같은 이미지를 중복 처리하지 않는다
        self.latest_image_msg = None

        if self.latest_scan is None:

            self.get_logger().warn("아직 /scan 데이터 없음")

            return


        try:

            if self.use_compressed:

                bgr = compressed_to_bgr(msg)

            else:

                bgr = nv21_to_bgr(msg)

        except Exception as e:

            self.get_logger().error(f"이미지 변환 실패: {e}")

            return

        if bgr is None:
            return

        img_h, img_w = bgr.shape[:2]


        if (
            img_w != CALIBRATION_WIDTH
            or img_h != CALIBRATION_HEIGHT
        ):

            self.get_logger().warn(
                f"캘리브레이션 해상도 불일치: "
                f"{img_w}x{img_h}, "
                f"기준 {CALIBRATION_WIDTH}x{CALIBRATION_HEIGHT}"
            )

            return


        t0 = time.time()

        results = self.model.predict(
            source=bgr,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False
        )

        infer_ms = (time.time() - t0) * 1000

        scan = self.latest_scan

        n_det = 0


        for result in results:

            for box in result.boxes:

                n_det += 1

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                conf = float(box.conf[0])

                label = self.target_classes[int(box.cls[0])]

                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0

                cam_angle_deg = self.pixel_to_angle_deg(u)


                lidar_angle_rad, r = lidar_lookup(
                    scan,
                    cam_angle_deg,
                    self.beam_window
                )

                lidar = None

                if r is not None:

                    lx = r * math.cos(lidar_angle_rad)
                    ly = r * math.sin(lidar_angle_rad)

                    lidar = {
                        "x": lx,
                        "y": ly,
                        # 카메라 기준 값에만 오프셋 보정
                        "z_cam": lx - CAMERA_FORWARD_OFFSET_M,
                    }


                boxest = box_estimate(
                    u,
                    y1,
                    y2,
                    img_h,
                    self.heights_cm.get(label)
                )

                height_cm = self.heights_cm.get(label)

                lidar_blocked = (
                    height_cm is not None
                    and height_cm < self.lidar_height_cm
                )

                src, chosen, note = choose_estimate(
                    lidar,
                    boxest,
                    self.mismatch_ratio,
                    lidar_blocked
                )

                head = (
                    f"[{label}] conf={conf:.2f} "
                    f"center_px=({u:.0f},{v:.0f}) "
                    f"cam_angle={cam_angle_deg:+.2f}deg"
                )

                detail = ""

                if lidar is not None:
                    detail += f" | LiDAR Zcam={lidar['z_cam'] * 100:.1f}cm"

                if boxest is not None:
                    detail += f" | 박스 Zcam={boxest['z_cam'] * 100:.1f}cm"

                if chosen is None:

                    self.get_logger().info(
                        f"{head} | 거리 없음 ({note}){detail}"
                    )

                    continue

                self.get_logger().info(
                    f"{head} src={src} "
                    f"Zcam={chosen['z_cam'] * 100:.1f}cm "
                    f"| 로봇 좌표 (x={chosen['x']:.2f}, y={chosen['y']:.2f})"
                    f"{detail}"
                    f"{' | ' + note if note else ''}"
                )


        if n_det == 0:

            self.get_logger().info("[탐지 결과] 객체 0개")

        self.get_logger().info(
            f"(추론 {infer_ms:.0f}ms, 탐지 {n_det}개)"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Camera + LiDAR Object Localization v4"
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=["cup"],
        help="탐지할 객체 클래스"
    )

    parser.add_argument(
        "--heights",
        nargs="*",
        default=[],
        help="물체 실제 높이(cm). 예: cup=10 bottle=16 book=28"
    )

    parser.add_argument("--conf", type=float, default=0.15)

    parser.add_argument("--hz", type=float, default=2.0)

    parser.add_argument("--imgsz", type=int, default=320)

    parser.add_argument("--compressed", action="store_true")

    parser.add_argument(
        "--mismatch-ratio",
        type=float,
        default=1.25,
        help="LiDAR 거리가 박스 추정의 이 배수보다 멀면 박스 사용"
    )

    parser.add_argument(
        "--beam-window",
        type=int,
        default=1,
        help="LiDAR 조회 시 좌우로 함께 볼 빔 개수"
    )

    parser.add_argument(
        "--lidar-height",
        type=float,
        default=LIDAR_PLANE_HEIGHT_CM,
        help="LiDAR 스캔 평면 높이(cm). 이보다 낮은 물체는 박스 거리를 사용"
    )

    parser.add_argument(
        "--image-topic",
        default=None,
        help="이미지 토픽 이름. 예: --compressed --image-topic /camera/relay/compressed"
    )

    parser.add_argument(
        "--qwen",
        action="store_true",
        help="터미널에서 한국어 명령을 받아 탐지 대상을 바꿈 (Ollama 필요)"
    )

    parser.add_argument(
        "--qwen-model",
        default="qwen2.5:0.5b",
        help="Ollama 모델 이름"
    )

    parser.add_argument(
        "--no-qwen-fallback",
        action="store_true",
        help="키워드 규칙만 쓰고 qwen 호출은 하지 않음"
    )

    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = ObjectLocalizerNode(
        args.classes,
        args.conf,
        args.hz,
        args.imgsz,
        args.compressed,
        parse_heights(args.heights),
        args.mismatch_ratio,
        args.beam_window,
        args.lidar_height,
        args.image_topic
    )

    if args.qwen:

        threading.Thread(
            target=node.command_loop,
            args=(not args.no_qwen_fallback, args.qwen_model),
            daemon=True
        ).start()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()