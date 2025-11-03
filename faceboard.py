import cv2
import mediapipe as mp
import math
import time
import pyautogui


CAM_WIDTH = 1280
CAM_HEIGHT = 720
MOUSE_STEP = 20
MOUTH_OPEN_RATIO_THRESHOLD = 0.035
ACTIVATION_COOLDOWN = 0.4

KEY_WIDTH = 26
KEY_HEIGHT = 26
KEY_SPACING_X = 5
KEY_SPACING_Y = 5
KEY_MARGIN_BOTTOM = 300
FONT = cv2.FONT_HERSHEY_SIMPLEX

QWERTY_ROWS = [
    list("1234567890"),
    list("qwertyuiop"),
    list("asdfghjkl"),
    list("zxcvbnm,.?")
]

SYMBOL_MAP = {
    '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
    '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
    'q': '[', 'w': ']', 'e': '{', 'r': '}', 't': '+',
    'y': '=', 'u': '-', 'i': '_', 'o': ':', 'p': ';',
    'a': '"', 's': "'", 'd': '<', 'f': '>', 'g': '/',
    'h': '\\', 'j': '|', 'k': '?', 'l': '~',
    'z': '`', 'x': '*', 'c': '!', 'v': '#', 'b': '$',
    'n': '%', 'm': '^', ',': ',', '.': '.', '?': '?'
}


class VirtualKey:
    def __init__(self, label, x, y, w, h):
        self.label = label
        self.rect = (int(x), int(y), int(w), int(h))

    def contains(self, px, py):
        x, y, w, h = self.rect
        return (x <= px <= x + w) and (y <= py <= y + h)


def build_keyboard(center_x, base_y):
    keys = []
    y = base_y

    for i, row in enumerate(QWERTY_ROWS):
        row_width = len(row) * (KEY_WIDTH + KEY_SPACING_X)
        x = center_x - row_width // 2

        # Row 2: SYM
        if i == 1:
            caps_align_x = x - 70
            keys.append(VirtualKey("SYM", caps_align_x, y, 60, KEY_HEIGHT))

        # Row 3: CAPS
        if i == 2:
            keys.append(VirtualKey("CAPS", x - 70, y, 60, KEY_HEIGHT))

        # Row 4: SHIFT
        if i == 3:
            keys.append(VirtualKey("SHIFT", x - 80, y, 75, KEY_HEIGHT))

        for ch in row:
            keys.append(VirtualKey(ch, x, y, KEY_WIDTH, KEY_HEIGHT))
            x += KEY_WIDTH + KEY_SPACING_X

        y += KEY_HEIGHT + KEY_SPACING_Y

    # Bottom row (SPACE, DELETE, ENTER)
    space_y = base_y + len(QWERTY_ROWS) * (KEY_HEIGHT + KEY_SPACING_Y)
    space_w = 180
    del_w = 90
    enter_w = 100
    spacing = 10

    total_bottom_width = space_w + del_w + enter_w + spacing * 2
    start_x = center_x - total_bottom_width // 2

    keys.append(VirtualKey("SPACE", start_x, space_y, space_w, KEY_HEIGHT))
    keys.append(VirtualKey("DELETE", start_x + space_w + spacing, space_y, del_w, KEY_HEIGHT))
    keys.append(VirtualKey("ENTER", start_x + space_w + del_w + spacing * 2, space_y, enter_w, KEY_HEIGHT))

    return keys


def build_overlapping_dpad_near_keyboard(keys):
    # Keyboard bounds
    max_x = max(k.rect[0] + k.rect[2] for k in keys)
    min_y = min(k.rect[1] for k in keys)
    max_y = max(k.rect[1] + k.rect[3] for k in keys)

    # D-pad positioning
    center_x = max_x + 5
    center_y = (min_y + max_y) // 2 - 40

    btn_w, btn_h = 30, 30
    gap = 5

    controls = {
        'UP': VirtualKey('^', center_x, center_y - btn_h - gap, btn_w, btn_h),
        'LEFT': VirtualKey('<', center_x - btn_w - gap, center_y, btn_w, btn_h),
        'LCLICK': VirtualKey('click', center_x, center_y, btn_w, btn_h),
        'RIGHT': VirtualKey('>', center_x + btn_w + gap, center_y, btn_w, btn_h),
        'DOWN': VirtualKey('v', center_x, center_y + btn_h + gap, btn_w, btn_h),
        'RCLICK': VirtualKey('Rclick', center_x, center_y + btn_h * 2 + 15, btn_w * 2, btn_h)
    }
    return controls




def draw_dpad(frame, controls, px, py):
    hovered = None
    for name, k in controls.items():
        x, y, w, h = k.rect
        is_hover = k.contains(px, py)

        color = (255, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if is_hover:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (100, 255, 100), -1)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            hovered = k

        ts = cv2.getTextSize(k.label, FONT, 0.45, 1)[0]
        cv2.putText(frame, k.label, 
                    (x + (w - ts[0]) // 2, y + (h + ts[1]) // 2 - 2),
                    FONT, 0.45, (255, 255, 255), 1)

    return frame, hovered

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def compute_mouth_open_ratio(lm, w, h):
    try:
        ux, uy = int(lm[13].x * w), int(lm[13].y * h)
        lx, ly = int(lm[14].x * w), int(lm[14].y * h)
        tx, ty = int(lm[10].x * w), int(lm[10].y * h)
        bx, by = int(lm[152].x * w), int(lm[152].y * h)
    except:
        return 0
    return math.hypot(ux - lx, uy - ly) / max(1.0, math.hypot(tx - bx, ty - by))


def compute_face_center(lm, w, h):
    nose = lm[1]
    return int(nose.x * w), int(nose.y * h)

# Main Loop
def main():
    pyautogui.FAILSAFE = False
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    last_mouth_state = False
    last_action = {}
    shift_active = False
    caps_active = False
    symbol_active = False
    keys, controls = [], {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if not keys:
            keys = build_keyboard(w // 2 - 100, h - KEY_MARGIN_BOTTOM)
            controls = build_overlapping_dpad_near_keyboard(keys)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        px, py = (-9999, -9999)
        mouth_ratio = 0.0
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            px, py = compute_face_center(lm, w, h)
            mouth_ratio = compute_mouth_open_ratio(lm, w, h)
            cv2.circle(frame, (px, py), 7, (0, 220, 0), -1)

        visual_upper = caps_active or shift_active
        hovered_key = None

        for k in keys:
            x, y, kw, kh = k.rect
            label = k.label

            display_label = label
            if len(display_label) == 1:
                if symbol_active and display_label.lower() in SYMBOL_MAP:
                    display_label = SYMBOL_MAP[display_label.lower()]
                elif display_label.isalpha():
                    display_label = display_label.upper() if visual_upper else display_label.lower()

            is_hover = k.contains(px, py)
            if is_hover:
                hovered_key = k
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + kw, y + kh), (100, 255, 100), -1)
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            # color
            color = (255, 255, 255)
            if k.label == "SHIFT" and shift_active:
                color = (100, 255, 100)
            elif k.label == "CAPS" and caps_active:
                color = (100, 100, 255)
            elif k.label == "SYM" and symbol_active:
                color = (255, 150, 0)

            cv2.rectangle(frame, (x, y), (x + kw, y + kh), color, 2)
            ts = cv2.getTextSize(display_label, FONT, 0.6, 2)[0]
            cv2.putText(frame, display_label,
                        (x + (kw - ts[0]) // 2, y + (kh + ts[1]) // 2),
                        FONT, 0.6, (255, 255, 255), 2)

        frame, hovered_control = draw_dpad(frame, controls, px, py)

        mouth_open = mouth_ratio > MOUTH_OPEN_RATIO_THRESHOLD
        now = time.time()
        triggered = None

        if mouth_open and not last_mouth_state and hovered_key:
            label = hovered_key.label
            if now - last_action.get(label, 0) > ACTIVATION_COOLDOWN:
                last_action[label] = now

                if label == "SPACE":
                    pyautogui.press("space")
                elif label == "DELETE":
                    pyautogui.press("backspace")
                elif label == "ENTER":
                    pyautogui.press("enter")
                elif label == "SHIFT":
                    shift_active = True
                elif label == "CAPS":
                    caps_active = not caps_active
                elif label == "SYM":
                    symbol_active = not symbol_active
                else:
                    ch = label
                    if len(ch) == 1:
                        if symbol_active and ch.lower() in SYMBOL_MAP:
                            actual_char = SYMBOL_MAP[ch.lower()]
                        elif ch.isalpha():
                            effective_upper = (caps_active and not shift_active) or (shift_active and not caps_active)
                            actual_char = ch.upper() if effective_upper else ch.lower()
                        else:
                            actual_char = ch
                    else:
                        actual_char = ch

                    try:
                        pyautogui.write(actual_char)
                    except Exception:
                        pyautogui.press(actual_char)

                    if shift_active:
                        shift_active = False

                triggered = f"Key: {label}"

        if mouth_open and hovered_control:
            cname = hovered_control.label
            if now - last_action.get(cname, 0) > 0.05:
                last_action[cname] = now
                
                scale = min(max((mouth_ratio - 0.03) * 15, 1), 5)
                dynamic_step = int(MOUSE_STEP * scale)

                if cname == '^': pyautogui.moveRel(0, -dynamic_step)
                elif cname == 'v': pyautogui.moveRel(0, dynamic_step)
                elif cname == '<': pyautogui.moveRel(-dynamic_step, 0)
                elif cname == '>': pyautogui.moveRel(dynamic_step, 0)
                elif cname.lower() == 'click': pyautogui.click(button='left')
                elif cname.lower() == 'rclick': pyautogui.click(button='right')

                triggered = f"{cname} ({dynamic_step})"



        if triggered:
            cv2.putText(frame, triggered, (w - 260, 40),
                        FONT, 0.8, (10, 250, 10), 2)

        last_mouth_state = mouth_open
        cv2.imshow("Face Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
