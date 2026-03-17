import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time
import os
import threading
import math
from collections import deque

# Suppress logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Model Definition (V5 Ultimate Architecture) ---
class NarutoNet(nn.Module):
    def __init__(self, num_classes):
        super(NarutoNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(207, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

# --- Feature Extraction ---
def get_finger_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))) / 180.0

def get_hand_features(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    base_lm = hand_landmarks.landmark[9]
    scale = math.sqrt((base_lm.x - wrist.x)**2 + (base_lm.y - wrist.y)**2 + (base_lm.z - wrist.z)**2)
    if scale < 0.0001: scale = 0.01
    f = []
    for lm in hand_landmarks.landmark:
        f.extend([(lm.x - wrist.x)/scale, (lm.y - wrist.y)/scale, (lm.z - wrist.z)/scale])
    fingers = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    for finger in fingers:
        f.append(get_finger_angle(hand_landmarks.landmark[finger[0]], hand_landmarks.landmark[finger[1]], hand_landmarks.landmark[finger[2]]))
        f.append(get_finger_angle(hand_landmarks.landmark[finger[1]], hand_landmarks.landmark[finger[2]], hand_landmarks.landmark[finger[3]]))
        f.append(get_finger_angle(hand_landmarks.landmark[finger[2]], hand_landmarks.landmark[finger[3]], hand_landmarks.landmark[finger[4]]))
    v1 = np.array([hand_landmarks.landmark[5].x - wrist.x, hand_landmarks.landmark[5].y - wrist.y, hand_landmarks.landmark[5].z - wrist.z])
    v2 = np.array([hand_landmarks.landmark[17].x - wrist.x, hand_landmarks.landmark[17].y - wrist.y, hand_landmarks.landmark[17].z - wrist.z])
    norm = np.cross(v1, v2)
    if np.linalg.norm(norm) > 0: norm = norm / np.linalg.norm(norm)
    f.extend(norm.tolist())
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            t1, t2 = hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[j]]
            f.append(math.sqrt((t1.x-t2.x)**2 + (t1.y-t2.y)**2) / scale)
    return f

# --- Jutsu Definitions ---
# Types: SEQ, MOTION, STATIC, HYBRID
# Positions: HAND, CENTER, FULL_SCREEN
JUTSU_MENU = {
    '1': {'name': 'Water Dragon', 'type': 'SEQ', 'seq': ['snake', 'ram', 'monkey', 'boar', 'horse', 'tiger']},
    '2': {'name': 'Earth Wall', 'type': 'SEQ', 'seq': ['tiger', 'hare', 'boar', 'dog'], 'video': 'earth', 'pos': 'CENTER'},
    '3': {'name': 'Dragon Flame', 'type': 'SEQ', 'seq': ['snake', 'dragon', 'hare', 'tiger']},
    '4': {'name': 'Chidori', 'type': 'SEQ', 'seq': ['ox', 'hare', 'monkey'], 'video': 'chidori', 'pos': 'HAND'},
    '5': {'name': 'Fireball', 'type': 'SEQ', 'seq': ['snake', 'ram', 'monkey', 'boar', 'horse', 'tiger'], 'video': 'fireball', 'pos': 'FULL_SCREEN'},
    '6': {'name': 'Phoenix Fire', 'type': 'SEQ', 'seq': ['rat', 'tiger', 'dog', 'ox', 'hare', 'tiger'], 'video': 'phoenix', 'pos': 'FULL_SCREEN'},
    '7': {'name': 'Wind Bullet', 'type': 'HYBRID', 'seal': 'tiger', 'motion': 'head_forward'},
    '8': {'name': 'Rasengan', 'type': 'MOTION', 'motion': 'circle', 'video': 'rasengan', 'pos': 'RASENGAN_MODE'},
    '9': {'name': 'Summoning', 'type': 'SEQ', 'seq': ['boar', 'dog', 'bird', 'monkey', 'ram'], 'video': 'summoning', 'pos': 'FULL_SCREEN'}
}

# --- Threaded Video Stream ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    def start(self): 
        if self.grabbed: threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
    def update(self):
        while not self.stopped:
            (self.grabbed, f) = self.stream.read()
            if self.grabbed: self.frame = f
    def read(self): return self.frame
    def stop(self): self.stopped = True; self.stream.release()

# --- Motion Analysis ---
class ShinobiAnalyzer:
    def __init__(self):
        self.left_wrist_history = deque(maxlen=30)
        self.right_wrist_history = deque(maxlen=30)
        
    def detect_circle(self, centers):
        if len(centers) < 15: return False
        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]
        # Skip if all zeros (no data)
        if sum(x_coords) == 0: return False
        w, h = max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)
        if w < 60 or h < 60: return False
        ratio = min(w, h) / max(w, h)
        return ratio > 0.4

# --- Chroma Key Video Handler ---
class ChromaKeyVideo:
    def __init__(self, video_path, loop_pos_sec=None, pause_at_end=False):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30
        self.loop_frame = int(loop_pos_sec * self.fps) if loop_pos_sec else 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pause_at_end = pause_at_end
        self.finished = False
        
    def get_frame(self, target_size=None):
        if self.finished and self.pause_at_end:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.total_frames - 2))
            ret, frame = self.cap.read()
        else:
            ret, frame = self.cap.read()
            if not ret:
                if self.pause_at_end:
                    self.finished = True
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.total_frames - 2))
                    ret, frame = self.cap.read()
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.loop_frame)
                    ret, frame = self.cap.read()
            
            curr_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if curr_frame >= self.total_frames - 1:
                if self.pause_at_end: self.finished = True
                else: self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.loop_frame)

        if target_size and frame is not None: frame = cv2.resize(frame, target_size)
        return frame

    def apply(self, background, pos, size=(300, 300), full_screen=False):
        if full_screen:
            size = (background.shape[1], background.shape[0])
            pos = (background.shape[1]//2, background.shape[0]//2)
        
        # ENSURE SQUARE SIZE FOR ROUND SHAPE
        if not full_screen:
            side = max(size)
            size = (side, side)
            
        vf = self.get_frame(target_size=size)
        if vf is None: return background
        hsv = cv2.cvtColor(vf, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        mask_inv = cv2.bitwise_not(mask)
        x, y = pos
        y_s, y_e = max(0, y - size[1]//2), min(background.shape[0], y + size[1]//2)
        x_s, x_e = max(0, x - size[0]//2), min(background.shape[1], x + size[0]//2)
        
        fh, fw = y_e - y_s, x_e - x_s
        if fh <= 0 or fw <= 0: return background
        
        vf_part = vf[:fh, :fw]
        mask_part = mask[:fh, :fw]
        mask_inv_part = mask_inv[:fh, :fw]

        roi = background[y_s:y_e, x_s:x_e]
        fg = cv2.bitwise_and(vf_part, vf_part, mask=mask_inv_part)
        bg = cv2.bitwise_and(roi, roi, mask=mask_part)
        background[y_s:y_e, x_s:x_e] = cv2.add(fg, bg)
        return background

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.finished = False

def run_app():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckp = torch.load('models/naruto_model_gpu.pth', map_location=device)
        classes = ckp['classes']
        model = NarutoNet(ckp['num_classes']).to(device); model.load_state_dict(ckp['model_state_dict']); model.eval()
    except: return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hand_engine = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    
    vs = VideoStream().start()
    analyzer = ShinobiAnalyzer()
    
    current_key = None
    seq_ptr = 0
    is_playing_fx = False
    pred_buf = deque(maxlen=5)
    
    # Rasengan Persistent States
    l_rot_locked = False
    r_rot_locked = False
    big_rasengan_active = False
    big_rasengan_time = 0
    
    # Load FX
    fx_dict = {
        'chidori': ChromaKeyVideo(os.path.join("animations", "chidori", "_efecto_CHIDORI_de_sasuke_pantalla_verde_green_screen_NARUTO_effect_con_SONIDO_1080p.mp4")),
        'rasengan': ChromaKeyVideo(os.path.join("animations", "rasengan", "Rasengan_green_screen_720p.mp4"), loop_pos_sec=2.0),
        'summoning': ChromaKeyVideo(os.path.join("animations", "summoning", "SUMMONING_JUTSU_GREEN_SCREEN_COPYRIGHT_FRRR_720P.mp4"), pause_at_end=True),
        'fireball': ChromaKeyVideo(os.path.join("animations", "fireball", "Explosion_croma_key_green_screen_720p.mp4")),
        'phoenix': ChromaKeyVideo(os.path.join("animations", "phoenix fire", "Fireball_Explosion_Green_Screen_DOWNLOAD_LINK_HD_Quality_Non_Copyright_1080P.mp4")),
        'earth': ChromaKeyVideo(os.path.join("animations", "earth wall", "Naruto_GreenScreen_Earth_Wall_Jutsu_720P.mp4"), pause_at_end=True)
    }

    while True:
        frame = vs.read()
        if frame is None: break
        frame = cv2.flip(frame, 1)
        h_f, w_f, _ = frame.shape
        
        h_res = hand_engine.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detected_sign = "zero"
        hand_centers = {"Left": None, "Right": None}
        
        if h_res.multi_hand_landmarks:
            slots = {"Left": [0.0]*91, "Right": [0.0]*91}
            for i, lm in enumerate(h_res.multi_hand_landmarks):
                lbl = h_res.multi_handedness[i].classification[0].label
                slots[lbl] = get_hand_features(lm)
                hand_centers[lbl] = (int(lm.landmark[9].x * w_f), int(lm.landmark[9].y * h_f))
                if lbl == "Left": analyzer.left_wrist_history.append(hand_centers[lbl])
                else: analyzer.right_wrist_history.append(hand_centers[lbl])
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            
            with torch.no_grad():
                inp = torch.tensor([slots["Left"] + slots["Right"] + [0.0]*25], dtype=torch.float32).to(device)
                prob, idx = torch.max(torch.softmax(model(inp), 1), 1)
                if prob.item() > 0.85: pred_buf.append(classes[idx.item()])
                else: pred_buf.append("zero")
            detected_sign = max(set(pred_buf), key=list(pred_buf).count) if pred_buf else "zero"
        else:
            analyzer.left_wrist_history.append((0,0)); analyzer.right_wrist_history.append((0,0))

        key = cv2.waitKey(1) & 0xFF
        
        if is_playing_fx:
            jutsu = JUTSU_MENU[current_key]
            fx = fx_dict.get(jutsu.get('video'))
            if fx:
                pos_type = jutsu.get('pos')
                if pos_type == 'FULL_SCREEN':
                    frame = fx.apply(frame, (0,0), full_screen=True)
                elif pos_type == 'CENTER':
                    frame = fx.apply(frame, (w_f//2, h_f - 200), size=(w_f, 500))
                elif pos_type == 'RASENGAN_MODE':
                    # Update Rotation Flags
                    if not l_rot_locked: l_rot_locked = analyzer.detect_circle(list(analyzer.left_wrist_history))
                    if not r_rot_locked: r_rot_locked = analyzer.detect_circle(list(analyzer.right_wrist_history))
                    
                    if big_rasengan_active:
                        # STAGE 3: MERGED BIG RASENGAN
                        elapsed = time.time() - big_rasengan_time
                        if elapsed > 4.5: # AUTO EXIT
                            is_playing_fx = False; current_key = None; big_rasengan_active = False
                            l_rot_locked = False; r_rot_locked = False; fx.reset()
                        else:
                            # Zoom effect: starts from center and expands
                            mid_x = (hand_centers["Left"][0] + hand_centers["Right"][0]) // 2 if (hand_centers["Left"] and hand_centers["Right"]) else w_f//2
                            mid_y = (hand_centers["Left"][1] + hand_centers["Right"][1]) // 2 if (hand_centers["Left"] and hand_centers["Right"]) else h_f//2
                            zoom = int(400 + (elapsed * 300)) # Expanding sphere
                            frame = fx.apply(frame, (mid_x, mid_y), size=(zoom, zoom))
                    
                    elif l_rot_locked and r_rot_locked:
                        # STAGE 2: TRIGGER MERGE
                        big_rasengan_active = True
                        big_rasengan_time = time.time()
                    else:
                        # STAGE 1: INDIVIDUAL ROTATIONS
                        if l_rot_locked and hand_centers["Left"]:
                            frame = fx.apply(frame, hand_centers["Left"], size=(280, 280))
                        if r_rot_locked and hand_centers["Right"]:
                            frame = fx.apply(frame, hand_centers["Right"], size=(280, 280))
                
                elif pos_type == 'HAND':
                    for c in filter(None, hand_centers.values()): frame = fx.apply(frame, c, size=(350, 350))
            
            if key == 27: 
                is_playing_fx = False; current_key = None; big_rasengan_active = False
                l_rot_locked = False; r_rot_locked = False; fx.reset() if fx else None
        
        elif current_key is None:
            # Compact sidebar
            cv2.rectangle(frame, (5, 5), (200, 420), (10, 10, 10), -1)
            cv2.putText(frame, "SCROLLS", (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0,255,255), 1)
            for i, (k,v) in enumerate(JUTSU_MENU.items()):
                cv2.putText(frame, f"[{k}] {v['name'][:12]}", (15, 60 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            if key in [ord(k) for k in JUTSU_MENU.keys()]: current_key = chr(key); seq_ptr = 0
            
        else:
            jutsu = JUTSU_MENU[current_key]
            cv2.putText(frame, jutsu['name'].upper(), (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            if jutsu['type'] == 'SEQ':
                for i, s in enumerate(jutsu['seq']):
                    c = (0,255,0) if i < seq_ptr else (255,255,255)
                    if i == seq_ptr: c = (0,165,255)
                    cv2.putText(frame, s[0].upper(), (230+i*60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)
                if seq_ptr < len(jutsu['seq']) and detected_sign == jutsu['seq'][seq_ptr]:
                    seq_ptr += 1
                    if seq_ptr == len(jutsu['seq']):
                        if 'video' in jutsu: is_playing_fx = True
                        else: current_key = None
            elif jutsu['type'] == 'MOTION':
                if analyzer.detect_circle(list(analyzer.left_wrist_history)) or analyzer.detect_circle(list(analyzer.right_wrist_history)):
                    is_playing_fx = True
            if key == 27: current_key = None

        cv2.imshow("SHINOBI ENGINE", frame)
        if key == ord('q'): break

    vs.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_app()
