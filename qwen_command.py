#!/usr/bin/env python3
"""
qwen_command.py

한국어 명령("컵 찾아줘") -> YOLO-World에 넣을 영어 클래스("cup")

1단계: 한국어 키워드 규칙
2단계: 규칙에 걸리지 않으면 Ollama의 Qwen 2.5 모델에 질문
       (허용 목록의 단어가 정확히 하나일 때만 인정하고, 아니면 unknown)

한 글자 키워드(컵, 병, 책, 잔, 공)는 "병원", "책임", "공부"처럼 다른 단어의
일부일 때 오탐이 나지 않도록 키워드 뒤에 오는 글자(조사, 명령어 시작)를 확인한다.

단독 테스트:
  python3 qwen_command.py "컵 찾아줘"
  python3 qwen_command.py --no-qwen "책상 위에 있는 책 찾아줘"
"""

import argparse
import json
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b"
TIMEOUT_S = 15

# 허용 클래스. 새 물체를 추가하려면 여기와 KEYWORDS에 함께 추가한다.
ALLOWED = ["cup", "bottle", "book", "chair", "person", "table",
           "backpack", "suitcase", "umbrella", "couch",
           "potted plant", "teddy bear", "sports ball", "bench"]

KEYWORDS = {
    "cup": ["컵", "머그", "텀블러", "잔"],
    "bottle": ["물병", "병", "보틀", "페트"],
    "book": ["책", "도서", "교재"],
    "chair": ["의자"],
    "person": ["사람", "사람들"],
    "table": ["테이블", "탁자", "책상", "식탁"],
    "backpack": ["가방", "배낭", "백팩", "책가방"],
    "suitcase": ["캐리어", "여행가방", "트렁크"],
    "umbrella": ["우산"],
    "couch": ["소파"],
    "potted plant": ["화분", "식물"],
    "teddy bear": ["곰인형", "인형"],
    "sports ball": ["축구공", "농구공", "야구공", "공"],
    "bench": ["벤치"],
}

# 한 글자 키워드 뒤에 올 수 있는 글자(조사, 명령어 시작)
SUFFIXES = (
    "이", "가", "은", "는", "을", "를", "도", "좀", "만", "에", "의", "와", "과",
    "랑", "로", "으로", "하고", "찾", "어디", "어딨", "있", "보", "줘", "주",
)

PROMPT = (
    "Convert the Korean command into exactly one English object word.\n"
    "Allowed words: " + ", ".join(ALLOWED) + ", unknown.\n"
    "Answer with the single word only.\n\n"
    "Command: 컵 찾아줘\nAnswer: cup\n"
    "Command: 물병 어디 있어\nAnswer: bottle\n"
    "Command: 책 좀 찾아줘\nAnswer: book\n"
    "Command: 의자 있는 곳으로 가\nAnswer: chair\n"
    "Command: 가방 어딨어\nAnswer: backpack\n"
    "Command: 오늘 날씨 어때\nAnswer: unknown\n"
    "Command: {command}\nAnswer:"
)


def _keyword_in(command, w):
    #키워드가 명령에 들어 있는지. 한 글자 키워드는 뒤 글자를 확인한다.
    if len(w) > 1:
        return w in command
    for tok in command.split():
        tok = tok.strip("?!.,~")
        start = 0
        while True:
            i = tok.find(w, start)
            if i < 0:
                break
            after = tok[i + 1:]
            if after == "" or after.startswith(SUFFIXES):
                return True
            start = i + 1
    return False


def match_keyword(command):
    #가장 긴 키워드가 일치한 클래스를 반환한다. (예: 책상은 book이 아니라 table)
    best, best_len = None, 0
    for cls, words in KEYWORDS.items():
        for w in words:
            if len(w) > best_len and _keyword_in(command, w):
                best, best_len = cls, len(w)
    return best


def parse_answer(text):
    #모델 답변에서 허용 단어가 정확히 하나일 때만 반환.
    low = (text or "").strip().lower()
    if low in ALLOWED:
        return low
    found = {w for w in ALLOWED if re.search(rf"\b{w}\b", low)}
    if len(found) == 1:
        return found.pop()
    return None


def ask_qwen(command, model=OLLAMA_MODEL, url=OLLAMA_URL, timeout=TIMEOUT_S):
    body = json.dumps({
        "model": model,
        "prompt": PROMPT.format(command=command),
        "stream": False,
        "options": {"temperature": 0, "num_predict": 8},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def get_object_name(command, use_qwen=True, model=OLLAMA_MODEL):
    #(클래스 또는 None, 출처: rule/qwen/none, 메모)를 반환한다.
    command = (command or "").strip()
    if not command:
        return None, "none", "빈 입력"

    cls = match_keyword(command)
    if cls:
        return cls, "rule", ""

    if not use_qwen:
        return None, "none", "키워드에 없음"

    try:
        answer = ask_qwen(command, model=model)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, "none", f"Ollama 연결 실패: {e} (ollama serve 실행 중인지 확인)"
    except Exception as e:  # noqa: BLE001
        return None, "none", f"qwen 호출 오류: {e}"

    cls = parse_answer(answer)
    if cls:
        return cls, "qwen", f"qwen 답변: {answer.strip()!r}"
    return None, "none", f"qwen 답변을 해석 못 함: {answer.strip()!r}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("--no-qwen", action="store_true")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    a = ap.parse_args()
    print(get_object_name(a.command, use_qwen=not a.no_qwen, model=a.model))