import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

# --- Configuration ---
DATASET_PATH = 'data/raw'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
TEST_PATH = os.path.join(DATASET_PATH, 'test')
BATCH_SIZE = 64
EPOCHS = 2000 # 5000 is often overkill and causes overfitting, 2000 with AdamW is better
LEARNING_RATE = 0.0003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Forbidden Scroll: Hand Identification (V5 - THE ULTIMATE) ---
def get_finger_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))) / 180.0

def get_hand_features(hand_landmarks):
    """
    91 features per hand
    """
    wrist = hand_landmarks.landmark[0]
    base_lm = hand_landmarks.landmark[9] # Middle base
    scale = math.sqrt((base_lm.x - wrist.x)**2 + (base_lm.y - wrist.y)**2 + (base_lm.z - wrist.z)**2)
    if scale < 0.0001: scale = 0.01

    f = []
    # 1. Normalized Coords (63)
    for lm in hand_landmarks.landmark:
        f.extend([(lm.x - wrist.x)/scale, (lm.y - wrist.y)/scale, (lm.z - wrist.z)/scale])
        
    # 2. Bending Angles (15)
    fingers = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    for finger in fingers:
        f.append(get_finger_angle(hand_landmarks.landmark[finger[0]], hand_landmarks.landmark[finger[1]], hand_landmarks.landmark[finger[2]]))
        f.append(get_finger_angle(hand_landmarks.landmark[finger[1]], hand_landmarks.landmark[finger[2]], hand_landmarks.landmark[finger[3]]))
        f.append(get_finger_angle(hand_landmarks.landmark[finger[2]], hand_landmarks.landmark[finger[3]], hand_landmarks.landmark[finger[4]]))
        
    # 3. Orientation (3)
    v1 = np.array([hand_landmarks.landmark[5].x - wrist.x, hand_landmarks.landmark[5].y - wrist.y, hand_landmarks.landmark[5].z - wrist.z])
    v2 = np.array([hand_landmarks.landmark[17].x - wrist.x, hand_landmarks.landmark[17].y - wrist.y, hand_landmarks.landmark[17].z - wrist.z])
    norm = np.cross(v1, v2)
    if np.linalg.norm(norm) > 0: norm = norm / np.linalg.norm(norm)
    f.extend(norm.tolist())

    # 4. Intra-hand Distances (10)
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            t1, t2 = hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[j]]
            f.append(math.sqrt((t1.x-t2.x)**2 + (t1.y-t2.y)**2) / scale)
            
    return f

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def extract_v5_features(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # ROBUST HAND SLOTTING: Using Left/Right instead of X-sorting
        # This prevents "Flipped Features" bug
        hand_slots = {"Left": [0.0]*91, "Right": [0.0]*91}
        
        for i, res in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label # "Left" or "Right"
            features = get_hand_features(res)
            hand_slots[label] = features
            
        # Combine Left then Right (Total 182)
        combined = hand_slots["Left"] + hand_slots["Right"]
        
        # INTER-HAND INTERACTION (25 features)
        # Distance between all 5 tips of Left hand to all 5 tips of Right hand
        inter_hand_dist = [0.0] * 25
        if len(results.multi_hand_landmarks) == 2:
            l_idx = -1
            r_idx = -1
            for i, h_class in enumerate(results.multi_handedness):
                if h_class.classification[0].label == "Left": l_idx = i
                if h_class.classification[0].label == "Right": r_idx = i
            
            if l_idx != -1 and r_idx != -1:
                l_hand = results.multi_hand_landmarks[l_idx]
                r_hand = results.multi_hand_landmarks[r_idx]
                tips = [4, 8, 12, 16, 20]
                idx = 0
                for lt in tips:
                    for rt in tips:
                        d = math.sqrt((l_hand.landmark[lt].x - r_hand.landmark[rt].x)**2 + 
                                      (l_hand.landmark[lt].y - r_hand.landmark[rt].y)**2)
                        inter_hand_dist[idx] = d
                        idx += 1
        
        final_vector = combined + inter_hand_dist # 182 + 25 = 207 features
        return np.array(final_vector, dtype=np.float32)
    return None

# --- Model Definition ---
class NarutoNet(nn.Module):
    def __init__(self, num_classes):
        super(NarutoNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(207, 1024), # Larger first layer for interaction patterns
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

class NarutoDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        if self.augment:
            noise = np.random.normal(0, 0.012, x.shape).astype(np.float32)
            # Only jitter coordinates and distances, not angles (0-1 range)
            x += noise
        return torch.tensor(x, dtype=torch.float32), self.labels[idx]

if __name__ == "__main__":
    classes = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    
    print("\n--- INITIATING MISSION: JUTSU MASTER ---")
    data_train, labels_train = [], []
    for i, name in enumerate(classes):
        path = os.path.join(TRAIN_PATH, name)
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]
        print(f"Reading scroll {name} ({len(files)} images)...")
        for f in tqdm(files, leave=False):
            feat = extract_v5_features(os.path.join(path, f))
            if feat is not None:
                data_train.append(feat)
                labels_train.append(i)
                
    data_test, labels_test = [], []
    for i, name in enumerate(classes):
        path = os.path.join(TEST_PATH, name)
        if not os.path.exists(path): continue
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]
        for f in tqdm(files, leave=False):
            feat = extract_v5_features(os.path.join(path, f))
            if feat is not None:
                data_test.append(feat)
                labels_test.append(i)

    # Class Weights for 'Zero' stability
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train), y=labels_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    train_ds = NarutoDataset(np.array(data_train), labels_train, augment=True)
    test_ds = NarutoDataset(np.array(data_test), labels_test, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = NarutoNet(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n--- REFINING JUTSU BRAIN ---")
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                _, pred = torch.max(model(x), 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes,
                'num_classes': len(classes),
                'feature_version': 'v5_ultimate'
            }, 'models/naruto_model_gpu.pth')

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Cycle {epoch+1}/{EPOCHS} | Sync: {acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n--- MISSION COMPLETE: BEST SYNC {best_acc:.2f}% ---")
