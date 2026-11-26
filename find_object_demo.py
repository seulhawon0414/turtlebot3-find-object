#!/usr/bin/env python3
"""
find_object_demo.py

"컵 찾아줘" 같은 한국어 명령을 받아 제자리에서 회전하며 물체를 찾고,
발견하면 정지하여 여러 프레임의 중앙값으로 물체의 위치(앞/좌우)와 거리를 출력한다.

- 명령 인식: qwen_command.get_object_name
- 거리 계산: qwen_object_localizer.py (LiDAR 우선, 낮은 물체는 박스 크기)
- 명령: "컵 찾아줘", "그만"/"멈춰", "앞으로 10", "뒤로 5" (cm 단위)

실행 예:
  python3 find_object_demo.py --compressed --image-topic /camera/relay/compressed --heights cup=9 --qwen-model qwen2.5:3b
  (--no-move: 로봇을 움직이지 않고 흐름만 확인)
"""

import argparse
import math
import random
import re
import statistics
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)

from sensor_msgs.msg import Image, CompressedImage, LaserScan
from geometry_msgs.msg import Twist, PointStamped
from ultralytics import YOLO

import qwen_object_localizer as base
from qwen_command import get_object_name

STOP_WORDS = ["멈춰", "그만", "정지", "stop"]
FORWARD_WORDS = ["앞으로", "전진", "직진"]
BACK_WORDS = ["뒤로", "후진"]
MOVE_DEFAULT_CM = 5.0     # 전진/후진 명령에서 거리를 생략하면 이동할 거리
MOVE_MAX_CM = 30.0        # 한 번에 이동할 수 있는 최대 거리
MOVE_STOP_M = 0.15        # 이동 방향 15cm 안에 장애물이 있으면 정지 

# LiDAR 거리와 박스 크기 거리의 비율 허용 범위
SIZE_RATIO_MIN = 0.6
SIZE_RATIO_MAX = 1.6
# 빔 대조 비율이 이 값 이상이면 경고 출력
CONTRAST_WARN = 0.95
# 박스 폭 바깥으로 이만큼 떨어진 양옆 빔을 사용 
CONTRAST_MARGIN_DEG = 4.0


def lidar_contrast(scan, cam_angle, half_width_deg, beam_window, r_center):
    """정면 빔 거리 / 박스 폭 바깥 양옆 빔 중 먼 쪽 거리. 계산할 수 없으면 None."""
    try:
        sides = []
        for sign in (+1.0, -1.0):
            ang = cam_angle + sign * (half_width_deg + CONTRAST_MARGIN_DEG)
            _, rr = base.lidar_lookup(scan, ang, beam_window)
            if rr is not None:
                sides.append(rr)
        if r_center is None or not sides:
            return None
        return r_center / max(sides)
    except Exception:  # noqa: BLE001
        return None


# ROS 없이 테스트할 수 있는 로직

def estimate_detection(scan, u, y1, y2, img_h, height_cm,
                       beam_window, mismatch_ratio, lidar_plane_cm,
                       cam_height_m=0.135, pitch_deg=0.0, half_width_deg=None):
    """박스 하나의 거리를 추정한다. LiDAR 우선, 낮은 물체는 박스 크기 사용."""

    import types  

    cam_angle = math.degrees(math.atan2(base.CX - u, base.FX))

    lidar_angle, r = base.lidar_lookup(scan, cam_angle, beam_window)
    lidar = None
    if r is not None:
        lx = r * math.cos(lidar_angle)
        ly = r * math.sin(lidar_angle)
        lidar = {"x": lx, "y": ly, "z_cam": lx - base.CAMERA_FORWARD_OFFSET_M}
    contrast = None
    if half_width_deg is not None:
        contrast = lidar_contrast(scan, cam_angle, half_width_deg, beam_window, r)

    boxest = base.box_estimate(u, y1, y2, img_h, height_cm)
    blocked = height_cm is not None and height_cm < lidar_plane_cm
    src, chosen, note = base.choose_estimate(
        lidar, boxest, mismatch_ratio, blocked
    )

    # LiDAR 거리와 박스 크기가 서로 맞지 않으면 거리를 버린다. 허용 범위는 SIZE_RATIO_MIN~MAX.
    if (not blocked and lidar is not None and boxest is not None
            and chosen is not None and src == "lidar"):
        ratio = lidar["z_cam"] / boxest["z_cam"]
        if ratio < SIZE_RATIO_MIN or ratio > SIZE_RATIO_MAX:
            chosen = None
            src = "none"
            note = (f"박스 크기와 LiDAR 거리가 맞지 않아 버림 "
                    f"(LiDAR/박스 = {ratio:.2f}, 허용 {SIZE_RATIO_MIN}~{SIZE_RATIO_MAX})")

    height = None
    if chosen is not None:
        height = base.height_estimate(
            chosen["z_cam"], y1, y2, cam_height_m, pitch_deg
        )

    return {
        "cam_angle": cam_angle,
        "src": src,
        "chosen": chosen,
        "note": note,
        "lidar": lidar,
        "box": boxest,
        "height": height,
        "contrast": contrast,
    }


def median_result(samples):
    #측정 샘플의 중앙값으로 최종 위치를 계산한다. 로봇 좌표: x 앞, y 왼쪽
    xs = [s["chosen"]["x"] for s in samples]
    ys = [s["chosen"]["y"] for s in samples]
    zs = [s["chosen"]["z_cam"] for s in samples]
    srcs = {}
    for s in samples:
        srcs[s["src"]] = srcs.get(s["src"], 0) + 1

    hs = [s["height"] for s in samples if s.get("height")]
    z_center = statistics.median([h["center"] for h in hs]) if hs else 0.0
    z_bottom = statistics.median([h["bottom"] for h in hs]) if hs else 0.0
    z_top = statistics.median([h["top"] for h in hs]) if hs else 0.0
    cs = [s["contrast"] for s in samples if s.get("contrast") is not None]
    contrast = statistics.median(cs) if cs else None

    return {
        "contrast": contrast,
        "n": len(samples),
        "x": statistics.median(xs),
        "y": statistics.median(ys),
        "z": z_center,          # 물체 중심의 바닥 기준 높이
        "z_bottom": z_bottom,   # 바닥에 놓인 물체면 0 근처 (검증용)
        "z_top": z_top,
        "z_cam": statistics.median(zs),
        "z_spread": (max(zs) - min(zs)) / 2.0,
        "conf": sum(s["conf"] for s in samples) / len(samples),
        "srcs": srcs,
    }


def format_result(target, res):
    side = "왼쪽" if res["y"] >= 0 else "오른쪽"
    bearing = abs(math.degrees(math.atan2(res["y"], res["x"])))
    src_txt = ", ".join(f"{k} {v}회" for k, v in res["srcs"].items())
    extra = ""
    ct = res.get("contrast")
    if ct is not None and "lidar" in res["srcs"]:
        extra = f"\n        LiDAR 빔 대조 비율 {ct:.2f} (정면 / 양옆 중 먼 쪽, 작을수록 물체에 맞음)"
        if ct >= CONTRAST_WARN:
            extra += (f"\n        [주의] 비율이 {CONTRAST_WARN} 이상이라 LiDAR가 물체를 놓치고 "
                      f"뒤쪽 벽을 읽었을 수 있습니다. 물체가 벽에 가깝거나 LiDAR 평면보다 "
                      f"낮으면 위 거리를 신뢰하기 어렵습니다.")
    return (
        f"[발견] {target}: 로봇 기준 3D 위치 (x={res['x'] * 100:.1f}, "
        f"y={res['y'] * 100:.1f}, z={res['z'] * 100:.1f}) cm  "
        f"[x 앞, y 왼쪽+, z 바닥 기준 위쪽]\n"
        f"        앞 {res['x'] * 100:.1f} cm, {side} {abs(res['y']) * 100:.1f} cm "
        f"(정면에서 {side} {bearing:.1f}°), 중심 높이 {res['z'] * 100:.1f} cm\n"
        f"        카메라 기준 거리 Zcam {res['z_cam'] * 100:.1f} cm "
        f"(±{res['z_spread'] * 100:.1f}), 측정 {res['n']}회, 출처: {src_txt}, "
        f"평균 conf {res['conf']:.2f}\n"
        f"        높이 검증값: 박스 아래 끝 {res['z_bottom'] * 100:.1f} cm "
        f"(바닥 물체면 0 근처), 위 끝 {res['z_top'] * 100:.1f} cm"
        f"{extra}"
    )


class SearchController:
    """IDLE -> SEARCH -> SETTLE -> MEASURE -> IDLE 상태 머신"""

    IDLE, SEARCH, SETTLE, MEASURE = "IDLE", "SEARCH", "SETTLE", "MEASURE"

    def __init__(self, max_search_s, settle_s, n_samples,
                 measure_timeout_s, min_conf, max_attempts=4,
                 cluster_deg=4.0, z_tol=0.25):
        self.max_search_s = max_search_s
        self.settle_s = settle_s
        self.n_samples = n_samples          # 일관된 샘플이 이만큼 모이면 결과
        self.measure_timeout_s = measure_timeout_s
        self.min_conf = min_conf
        self.max_attempts = max_attempts
        self.cluster_deg = cluster_deg      # 같은 물체로 볼 각도 차이
        self.z_tol = z_tol                  # 같은 물체로 볼 거리 차이

        self.state = self.IDLE
        self.target = None
        self.samples = []
        self.attempts = 0
        self.t_search = 0.0
        self.t_settle = 0.0
        self.t_measure = 0.0

    @property
    def wants_spin(self):
        return self.state == self.SEARCH

    def start(self, target, now):
        self.target = target
        self.state = self.SEARCH
        self.t_search = now
        self.attempts = 0
        self.samples = []

    def stop(self):
        self.state = self.IDLE

    def best_cluster(self):
    #오탐을 걸러내기 위해 각도와 거리가 비슷한 샘플끼리 묶어 가장 큰 묶음을 반환한다.
        best = []
        for s in self.samples:
            members = [
                t for t in self.samples
                if abs(t["cam_angle"] - s["cam_angle"]) <= self.cluster_deg
                and abs(t["chosen"]["z_cam"] - s["chosen"]["z_cam"])
                <= self.z_tol * s["chosen"]["z_cam"]
            ]
            if len(members) > len(best):
                best = members
        return best

    def update(self, now, det):
        #탐지 결과(det, 없으면 None)를 받아 상태를 갱신하고 발생한 이벤트 목록을 반환한다.
        ev = []

        if self.state == self.SEARCH:
            if now - self.t_search > self.max_search_s:
                self.state = self.IDLE
                ev.append(("fail", f"{self.max_search_s:.0f}초 동안 {self.target}를 못 찾았어요"))
            elif (det is not None and det["conf"] >= self.min_conf
                  and det["chosen"] is not None):
                self.state = self.SETTLE
                self.t_settle = now + self.settle_s
                ev.append(("found", det))

        elif self.state == self.SETTLE:
            if now >= self.t_settle:
                self.state = self.MEASURE
                self.samples = []
                self.t_measure = now
                ev.append(("measuring", None))

        elif self.state == self.MEASURE:
            if (det is not None and det["conf"] >= self.min_conf
                    and det["chosen"] is not None):
                self.samples.append(det)

            cluster = self.best_cluster()

            if len(cluster) >= self.n_samples:
                self.state = self.IDLE
                ev.append(("result", median_result(cluster)))

            elif now - self.t_measure > self.measure_timeout_s:
                if len(cluster) >= 2:
                    self.state = self.IDLE
                    ev.append(("result", median_result(cluster)))
                else:
                    self.attempts += 1
                    if self.attempts >= self.max_attempts:
                        self.state = self.IDLE
                        ev.append(("fail", "발견은 했지만 일관된 거리 측정값을 "
                                           "모으지 못했어요 "))
                    else:
                        self.state = self.SEARCH
                        self.t_search = now
                        ev.append(("retry", "거리 측정 실패, 다시 탐색합니다"))

        return ev



# ROS 노드

class FindObjectNode(Node):

    def __init__(self, args):
        super().__init__("find_object_demo")

        self.args = args
        self.heights_cm = base.parse_heights(args.heights)
        self.ctl = SearchController(
            args.max_search_s, args.settle_s, args.samples,
            args.measure_timeout_s, args.min_conf,
            max_attempts=args.max_attempts,
            cluster_deg=args.cluster_deg,
        )

        self.get_logger().info("YOLO-World 로딩 중")
        self.model = YOLO("yolov8s-world.pt")
        self.model.set_classes(["cup"])

        self.lock = threading.Lock()
        self.pending_cmd = None
        self.pending_target = None

        self.latest_scan = None
        self.latest_image_msg = None
        self.stop_ticks = 0
        self.pending_move_cm = None   # +앞 / -뒤 
        self.move_end = 0.0           # 이동 종료 시각
        self.move_sign = 0.0
        self.move_cm = 0.0
        self.move_auto = False        # 탐색 중 자동 이동인지
        self.search_rot = 0.0         # 탐색 시작 후 누적 회전각

        # 진단용 카운터
        self.n_image = 0
        self.n_scan = 0
        self.n_frames_run = 0   # 실제로 YOLO 를 돌린 프레임 수
        self.n_boxes = 0        # 박스가 하나라도 나온 프레임 수
        self.last_status = 0.0

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if args.compressed:
            self.create_subscription(
                CompressedImage,
                args.image_topic or "/camera/image_raw/compressed",
                self.image_callback, image_qos,
            )
        else:
            self.create_subscription(
                Image, args.image_topic or "/camera/image_raw",
                self.image_callback, image_qos,
            )

        self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.point_pub = self.create_publisher(PointStamped, "/target_position", 10)

        self.create_timer(0.1, self.control_callback)
        self.create_timer(1.0 / args.hz, self.infer_callback)

        self.get_logger().info(
            f"준비 완료 | 회전 {args.spin} rad/s, 최대 탐색 {args.max_search_s}초, "
            f"정지 후 {args.settle_s}초 대기, 측정 {args.samples}회, "
            f"min-conf {args.min_conf}, 높이(cm) {self.heights_cm}, "
            f"{'움직이지 않음(--no-move)' if args.no_move else '로봇 회전 사용'}"
        )

    # 센서

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.n_scan += 1
        if self.n_scan == 1:
            self.get_logger().info(f"[SENSOR] LiDAR 수신 성공 (포인트 {len(msg.ranges)}개)")

    def image_callback(self, msg):
        self.latest_image_msg = msg
        self.n_image += 1
        if self.n_image == 1:
            self.get_logger().info("[SENSOR] 카메라 이미지 수신 성공")

    # 명령 

    def command_loop(self):
        print("명령을 입력하세요. 예: 책 좀 찾아줘 / 그만. 종료는 Ctrl+C")
        use_qwen = not self.args.no_qwen_fallback

        while rclpy.ok():
            try:
                text = input("명령> ").strip()
            except EOFError:
                return
            if not text:
                continue

            if any(w in text for w in STOP_WORDS):
                with self.lock:
                    self.pending_cmd = "stop"
                print("중지합니다")
                continue

            is_fwd = any(w in text for w in FORWARD_WORDS)
            is_back = any(w in text for w in BACK_WORDS)
            if is_fwd or is_back:
                m = re.search(r"(\d+(?:\.\d+)?)", text)
                dist = float(m.group(1)) if m else MOVE_DEFAULT_CM
                dist = max(1.0, min(MOVE_MAX_CM, dist))
                with self.lock:
                    self.pending_cmd = "move"
                    self.pending_move_cm = dist if (is_fwd and not is_back) else -dist
                continue

            cls, src, note = get_object_name(
                text, use_qwen=use_qwen, model=self.args.qwen_model
            )
            if cls is None:
                print(f"알아듣지 못함 ({note})")
                continue

            if src == "qwen":
                ans = input(f"  qwen이 '{cls}'로 해석했어요. 찾을까요? (y/n) ").strip().lower()
                if ans not in ("y", "yes", "ㅇ", "네"):
                    continue

            msg = f"  -> '{cls}' 찾기 시작 (출처: {src})"
            if cls not in self.heights_cm:
                msg += " | 높이 정보 없음: LiDAR 값만 사용"
            print(msg)

            with self.lock:
                self.pending_cmd = "search"
                self.pending_target = cls

    def apply_commands(self):
        with self.lock:
            cmd, tgt = self.pending_cmd, self.pending_target
            self.pending_cmd = None

        if cmd == "stop":
            self.ctl.stop()
            self.move_end = 0.0
            self.stop_ticks = 10
            self.get_logger().info("중지됨")

        elif cmd == "move":
            self.start_move(self.pending_move_cm)

        elif cmd == "search":
            self.model.set_classes([tgt])
            self.latest_image_msg = None
            self.ctl.start(tgt, time.time())
            self.search_rot = 0.0
            self.get_logger().info(f"탐색 시작: {tgt}")

    #  앞뒤 이동 

    def sector_min(self, center_deg, half_deg=15.0):
        #이동 시 부딪히지 않도록 해당 방향(0=정면, 180=뒤)의 가장 가까운 거리(m)를 구한다.
        scan = self.latest_scan
        if scan is None:
            return None
        best = None
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r < max(scan.range_min, 0.03):
                continue
            a = math.degrees(scan.angle_min + i * scan.angle_increment)
            diff = (a - center_deg + 180.0) % 360.0 - 180.0
            if abs(diff) <= half_deg and (best is None or r < best):
                best = r
        return best

    def start_move(self, cm, auto=False):
        if self.args.no_move:
            if not auto:
                print("--no-move 상태라 이동하지 않아요")
            return
        if not auto and self.ctl.state != SearchController.IDLE:
            print("탐색/측정 중에는 이동할 수 없어요. 이동을 원하면 '그만'을 입력하세요")
            return
        sign = 1.0 if cm > 0 else -1.0
        near = self.sector_min(0.0 if sign > 0 else 180.0)
        if near is not None and near < MOVE_STOP_M + 0.05:
            if not auto:
                print(f"  -> {'앞' if sign > 0 else '뒤'} {near * 100:.0f}cm 안에 물체가 있어 이동하지 않아요")
            return
        self.move_sign = sign
        self.move_cm = abs(cm)
        self.move_auto = auto
        self.move_end = time.time() + abs(cm) / 100.0 / self.args.move_speed
        msg = (f"{'앞으로' if sign > 0 else '뒤로'} 약 {abs(cm):.0f}cm 이동"
               f"{' (한 바퀴 탐색 후 위치 이동)' if auto else ' (시간 제어 이동으로 오차가 있을 수 있음)'}")
        if auto:
            self.get_logger().info(msg)
        else:
            print("  -> " + msg)

    def auto_nudge(self):
        #한 바퀴 돌아도 못 찾았을 때: 앞/뒤 중 여유 있는 쪽으로 무작위로 조금 이동
        cm = random.uniform(self.args.nudge_min_cm, self.args.nudge_max_cm)
        need = cm / 100.0 + MOVE_STOP_M   # 이동할 거리 + 최소 안전거리
        cands = []
        for sign, ang in ((1.0, 0.0), (-1.0, 180.0)):
            near = self.sector_min(ang)
            if near is None or near > need:
                cands.append(sign)
        if not cands:
            return
        self.start_move(random.choice(cands) * cm, auto=True)

    def finish_move(self, why):
        auto = self.move_auto
        self.move_end = 0.0
        self.move_auto = False
        self.cmd_pub.publish(Twist())
        self.stop_ticks = 10
        if auto:
            self.get_logger().info(f"자동 이동 끝 ({why})")
            return
        front = self.sector_min(0.0, 3.0)
        txt = f" | 정면 LiDAR 거리 {front * 100:.1f}cm" if front is not None else ""
        print(f"이동 끝 ({why}){txt}")

    # 회전 명령 (10Hz)

    def control_callback(self):
        if self.args.no_move:
            return

        if self.move_end > 0.0:
            want = (SearchController.SEARCH if self.move_auto
                    else SearchController.IDLE)
            if self.ctl.state != want:
                self.finish_move("상태가 바뀌어 중단")
                return
            near = self.sector_min(0.0 if self.move_sign > 0 else 180.0)
            if near is not None and near < MOVE_STOP_M:
                self.finish_move(f"{'앞' if self.move_sign > 0 else '뒤'} {near * 100:.0f}cm에 물체가 있어 멈춤")
            elif time.time() >= self.move_end:
                self.finish_move("완료")
            else:
                tw = Twist()
                tw.linear.x = self.move_sign * self.args.move_speed
                self.cmd_pub.publish(tw)
            return

        if self.ctl.wants_spin:
            tw = Twist()
            tw.angular.z = self.args.spin
            self.cmd_pub.publish(tw)
            self.stop_ticks = 10
            self.search_rot += self.args.spin * 0.1
            if not self.args.no_nudge and self.search_rot >= 2.0 * math.pi:
                self.search_rot = 0.0
                self.auto_nudge()
        elif self.stop_ticks > 0:
            self.cmd_pub.publish(Twist())
            self.stop_ticks -= 1

    def halt(self):
        if not self.args.no_move:
            for _ in range(3):
                self.cmd_pub.publish(Twist())

    # 탐지 

    def detect_best(self):
        msg = self.latest_image_msg
        scan = self.latest_scan
        if msg is None or scan is None:
            return None
        self.latest_image_msg = None

        try:
            bgr = (base.compressed_to_bgr(msg) if self.args.compressed
                   else base.nv21_to_bgr(msg))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"이미지 변환 실패: {e}")
            return None
        if bgr is None:
            return None

        img_h, img_w = bgr.shape[:2]
        if img_w != base.CALIBRATION_WIDTH or img_h != base.CALIBRATION_HEIGHT:
            self.get_logger().warn(f"해상도 불일치: {img_w}x{img_h}")
            return None

        res = self.model.predict(
            source=bgr, conf=self.args.conf, imgsz=self.args.imgsz, verbose=False
        )[0]
        self.n_frames_run += 1
        if len(res.boxes) == 0:
            return None
        self.n_boxes += 1

        best = max(res.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        conf = float(best.conf[0])
        u = (x1 + x2) / 2.0
        half_width_deg = math.degrees(math.atan2((x2 - x1) / 2.0, base.FX))

        det = estimate_detection(
            scan, u, y1, y2, img_h,
            self.heights_cm.get(self.ctl.target),
            self.args.beam_window, self.args.mismatch_ratio,
            self.args.lidar_height,
            self.args.cam_height_cm / 100.0, self.args.cam_pitch_deg,
            half_width_deg=half_width_deg,
        )
        det["conf"] = conf
        det["u"] = u
        return det

    def infer_callback(self):
        self.apply_commands()
        st = self.ctl.state

        if st == SearchController.IDLE:
            self.latest_image_msg = None
            return

        now = time.time()
        if now - self.last_status > 5.0:
            self.last_status = now
            warn = ""
            if self.n_image == 0:
                warn = " | 이미지가 안 들어옴 (토픽/--compressed/--image-topic 확인)"
            elif self.n_scan == 0:
                warn = " | /scan 이 안 들어옴"
            self.get_logger().info(
                f"[상태] {st} | 이미지 수신 {self.n_image}, scan 수신 {self.n_scan}, "
                f"YOLO 실행 {self.n_frames_run}프레임, 박스 나온 프레임 {self.n_boxes}{warn}"
            )

        if st == SearchController.SETTLE:
            # 정지 직후의 흔들린 프레임은 버림
            self.latest_image_msg = None
            det = None
        else:
            det = self.detect_best()

        if det is not None and st == SearchController.SEARCH:
            self.get_logger().info(
                f"탐지 conf={det['conf']:.2f} 각도 {det['cam_angle']:+.1f}°"
                f"{' (확신 부족, 계속 탐색)' if det['conf'] < self.args.min_conf else ''}"
            )

        for kind, payload in self.ctl.update(time.time(), det):

            if kind == "found":
                self.get_logger().info(
                    f"발견! conf={payload['conf']:.2f} -> 정지 후 "
                    f"{self.args.settle_s}초 기다렸다가 측정합니다"
                )
            elif kind == "measuring":
                self.latest_image_msg = None
                self.get_logger().info(f"측정 중 ({self.args.samples}회)...")
            elif kind == "retry":
                self.get_logger().warn(payload)
            elif kind == "fail":
                self.get_logger().warn(payload)
                print(f"[실패] {payload}")
            elif kind == "result":
                print(format_result(self.ctl.target, payload))
                if payload["n"] < self.args.samples:
                    print(f"        (참고: 일관된 샘플이 {payload['n']}개뿐이라 "
                          f"신뢰도가 낮아요. 기준은 {self.args.samples}개)")
                self.publish_point(payload)

    def publish_point(self, res):
        m = PointStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_scan"
        m.point.x = float(res["x"])
        m.point.y = float(res["y"])
        m.point.z = float(res["z"])  # 물체 중심의 바닥 기준 높이(m)
        self.point_pub.publish(m)



# main

def main():
    ap = argparse.ArgumentParser(description="한국어 명령으로 물체 찾기 데모")

    ap.add_argument("--heights", nargs="*", default=[],
                    help="물체 실제 높이(cm). 예: cup=9 book=28 bottle=16")
    ap.add_argument("--compressed", action="store_true")
    ap.add_argument("--image-topic", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.08,
                    help="YOLO 탐지 하한 (낮게 두고 아래 min-conf 로 거름)")
    ap.add_argument("--min-conf", type=float, default=0.12,
                    help="이 값 이상이면 일단 멈추고 측정에 사용. 틀린 탐지는 "
                         "멈춘 뒤 '일관성 확인'(--samples, --cluster-deg)으로 거른다")
    ap.add_argument("--hz", type=float, default=3.0)

    ap.add_argument("--spin", type=float, default=0.3,
                    help="탐색 회전 속도(rad/s). 0.1~0.6 로 제한")
    ap.add_argument("--move-speed", type=float, default=0.05,
                    help="앞뒤 이동 속도(m/s). 0.02~0.1 로 제한")
    ap.add_argument("--no-nudge", action="store_true",
                    help="한 바퀴 돌아도 못 찾을 때 앞뒤로 자리를 옮기지 않음")
    ap.add_argument("--nudge-min-cm", type=float, default=10.0)
    ap.add_argument("--nudge-max-cm", type=float, default=20.0)
    ap.add_argument("--max-search-s", type=float, default=120.0)
    ap.add_argument("--settle-s", type=float, default=1.0,
                    help="정지 후 측정 전에 기다리는 시간(초)")
    ap.add_argument("--samples", type=int, default=3,
                    help="같은 각도/거리로 일관되게 나온 샘플이 이만큼 모이면 결과 출력")
    ap.add_argument("--measure-timeout-s", type=float, default=25.0)
    ap.add_argument("--max-attempts", type=int, default=4,
                    help="측정 확인에 실패했을 때 다시 탐색하는 최대 횟수")
    ap.add_argument("--cluster-deg", type=float, default=4.0,
                    help="같은 물체로 볼 각도 차이(deg)")

    ap.add_argument("--cam-height-cm", type=float, default=13.5,
                    help="카메라 렌즈 중심의 바닥 기준 높이(cm)")
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0,
                    help="카메라가 아래로 기울어진 각도(deg). 수평이면 0")

    ap.add_argument("--mismatch-ratio", type=float, default=1.25)
    ap.add_argument("--beam-window", type=int, default=1)
    ap.add_argument("--lidar-height", type=float, default=base.LIDAR_PLANE_HEIGHT_CM)

    ap.add_argument("--no-move", action="store_true",
                    help="cmd_vel 을 내보내지 않음 (손으로 돌리며 흐름만 확인)")
    ap.add_argument("--qwen-model", default="qwen2.5:0.5b")
    ap.add_argument("--no-qwen-fallback", action="store_true")

    args, ros_args = ap.parse_known_args()
    args.spin = max(0.1, min(0.6, args.spin))
    args.move_speed = max(0.02, min(0.1, args.move_speed))

    rclpy.init(args=ros_args)
    node = FindObjectNode(args)

    threading.Thread(target=node.command_loop, daemon=True).start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.halt()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()