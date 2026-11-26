# TurtleBot3 Find Object

**A TurtleBot3 system that finds an object from a Korean text command and reports its distance and 3D position, using YOLO-World, LiDAR, and box-size ranging.**

한국어 명령("내 물병 어딨지?, 내 가방 어딨어?")으로 TurtleBot3가 물체를 찾아 거리와 3D 위치를 알려 주는 시스템입니다.

<!-- 여기에 데모 영상 -->

## 동작 방식

1. **명령 인식**: 키워드 규칙으로 먼저 찾고, 해당하지 않으면 Ollama Qwen으로 물체 이름을 판단합니다.
2. **탐색**: 제자리에서 회전하며 YOLO-World로 물체를 찾습니다. 한 바퀴 돌아도 없으면 조금 이동한 뒤 다시 찾습니다.
3. **측정**: 물체를 찾으면 멈추고, LiDAR와 박스 크기로 거리를 구해 로봇 기준 3D 위치(x, y, z)를 계산합니다.
4. **알림**: 결과를 터미널에 출력하고 `/target_position` 토픽(PointStamped)으로 발행합니다.

![시스템 구조도]()

## 거리 추정

| 방법 | 평균 오차 (cm) |
|---|---|
| Depth Anything V2 (2차식 보정) | 13.26 |
| LiDAR | 3.14 |
| 박스 크기 | 2.41 |

- 기본은 LiDAR를 사용합니다.
- LiDAR 평면보다 낮은 물체는 LiDAR가 놓칠 수 있어, 박스 크기로 거리를 구합니다.
- 측정한 결과 책은 LiDAR로 67cm에서 103cm 사이에서 오차 +1.3cm에서 +6cm였고, 컵은 박스 크기로 38cm에서 43.5cm 사이에서 오차 2cm 이내였습니다.

## 좌표계

로봇 기준 x는 앞, y는 왼쪽(+), z는 바닥에서 물체 중심까지의 높이입니다.

## 파일

| 파일 | 역할 |
|---|---|
| `find_object_demo.py` | 실행 파일. 탐색·측정 흐름을 제어하고 아래 두 파일을 불러옵니다. |
| `qwen_command.py` | 한국어 명령에서 물체 이름 추출 |
| `qwen_object_localizer.py` | YOLO-World 탐지, 거리·3D 위치 계산 |

## 실행

환경: ROS 2 Humble, TurtleBot3 Burger, Ollama, YOLO-World

```bash
ollama pull qwen2.5:3b
python3 find_object_demo.py --qwen-model qwen2.5:3b
```

카메라 토픽이 압축 이미지면 `--compressed`, 토픽 이름이 다르면 `--image-topic`으로 지정합니다.

## 설정 값

코드 안의 아래 값을 환경에 맞게 바꿔야 합니다.

- 카메라 내부 파라미터 (FX, FY, CX, CY)
- 카메라와 LiDAR의 위치 차이
- LiDAR 높이(LiDAR 평면)
- 물체별 실제 높이 (박스 크기 거리 계산에 사용)

## 주의

- 박스 크기 방식은 물체의 실제 높이를 미리 알아야 합니다.
- z(높이)는 약간의 흔들림이 존재합니다.
