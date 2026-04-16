"""
================================================================================
Reconocimiento de caras sintéticas con Ridge Regression
Inspirado en An, Liu & Venkatesh (2007) — Face Recognition Using KRR
================================================================================
"""
import os 
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2 #TODO only import used stuff
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

RNG = np.random.default_rng(42)
IMG_SIZE = 32          # imágenes de 32x32 como en el paper
N_IDENTIDADES = 10
N_POR_IDENTIDAD = 100  # 1000 imágenes en total

DIR_NAME = "dataset" # Nombre del directiorio con las imagenes clasificadas

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (478-point model)
# ---------------------------------------------------------------------------
 
# Eyes — 6 landmarks each, ordered for EAR
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
 
# Eyebrows — 5 landmarks each
LEFT_BROW  = [336, 296, 334, 293, 300]
RIGHT_BROW = [107,  66, 105,  63,  70]
 
# Nose
NOSE_BRIDGE = 6
NOSE_TIP    = 4
 
# Mouth
MOUTH_LEFT       = 61
MOUTH_RIGHT      = 291
MOUTH_CENTER_TOP = 13
MOUTH_CENTER_BOT = 14
 
# Face bounding references
FACE_TOP   = 10
FACE_BOT   = 152
FACE_LEFT  = 234
FACE_RIGHT = 454

#Utilidades : TODO: QUITARLAS E INCORPORARLAS DE MANERA MAS LIMPA
def _lm_xy(landmarks: list, idx: int, w: int, h: int) -> np.ndarray:
    """Return pixel-space (x, y) for a single NormalizedLandmark."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)
 
 
def _eye_aspect_ratio(landmarks: list, eye_idx: list, w: int, h: int) -> float:
    """
    Eye Aspect Ratio (EAR) — Soukupová & Čech 2016.
        EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    Typical range: ~0.30 open eye, <0.20 closed eye.
    """
    p = [_lm_xy(landmarks, i, w, h) for i in eye_idx]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return float((A + B) / (2.0 * C + 1e-6))
 
 
def _brow_curvature(landmarks: list, brow_idx: list, w: int, h: int) -> float:
    """
    Fit a quadratic (x -> y) to the brow landmarks.
    Returns the leading coefficient, negated for screen-y convention:
      positive  ->  arched/raised brow
      negative  ->  flat or furrowed brow
    """
    pts = np.array([_lm_xy(landmarks, i, w, h) for i in brow_idx])
    coeffs = np.polyfit(pts[:, 0], pts[:, 1], 2)
    return float(-coeffs[0])   # negate: screen y grows downward
 
 
def _mouth_curvature(landmarks: list, w: int, h: int) -> float:
    """
    Measures vertical offset of mouth corners relative to lip centre-line.
      positive  ->  smile (corners above centre)
      negative  ->  frown (corners below centre)
    """
    left  = _lm_xy(landmarks, MOUTH_LEFT,       w, h)
    right = _lm_xy(landmarks, MOUTH_RIGHT,      w, h)
    top   = _lm_xy(landmarks, MOUTH_CENTER_TOP, w, h)
    bot   = _lm_xy(landmarks, MOUTH_CENTER_BOT, w, h)
    corner_y = (left[1] + right[1]) / 2.0
    centre_y = (top[1]  + bot[1])   / 2.0
    return float(centre_y - corner_y)

# ----------------------------------------------------------------------
# 1. Definición de las 10 identidades prototípicas
# ----------------------------------------------------------------------
# Cada fila = [cx_eyeL, cy_eyeL, cx_eyeR, cy_eyeR, r_eye,
#              cx_nose, cy_nose, len_nose,
#              cx_mouth, cy_mouth, w_mouth, curv_mouth]
PROTOTIPOS = np.array([
    [-0.20, 0.20,  0.20, 0.20, 0.06,   0.00, -0.02, 0.18,   0.00, -0.25, 0.22,  0.10],  # 0
    [-0.25, 0.22,  0.25, 0.22, 0.08,   0.00,  0.00, 0.22,   0.00, -0.28, 0.26, -0.10],  # 1
    [-0.15, 0.18,  0.15, 0.18, 0.05,   0.00, -0.05, 0.12,   0.00, -0.22, 0.18,  0.12],  # 2
    [-0.30, 0.25,  0.30, 0.25, 0.09,   0.00,  0.02, 0.20,   0.00, -0.30, 0.28,  0.00],  # 3
    [-0.18, 0.15,  0.22, 0.15, 0.06,   0.02, -0.04, 0.16,  -0.02, -0.20, 0.20,  0.08],  # 4
    [-0.22, 0.28,  0.22, 0.28, 0.07,   0.00,  0.05, 0.25,   0.00, -0.32, 0.24, -0.05],  # 5
    [-0.28, 0.20,  0.28, 0.20, 0.10,   0.00, -0.08, 0.10,   0.00, -0.18, 0.30,  0.14],  # 6
    [-0.20, 0.25,  0.20, 0.25, 0.05,   0.00,  0.00, 0.20,   0.00, -0.26, 0.16, -0.12],  # 7
    [-0.24, 0.18,  0.24, 0.18, 0.08,  -0.02, -0.03, 0.14,   0.02, -0.24, 0.22,  0.06],  # 8
    [-0.16, 0.22,  0.16, 0.22, 0.06,   0.00,  0.04, 0.18,   0.00, -0.28, 0.20, -0.08],  # 9
])

def extract_params(img):
    """
    Extrae parmetros para la ia de una imagen de Opencv
    regresa un arreglo de 10 elementos (los definidos para
    reconocimiento facial
    """
    #TODO HAcerla xd

    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )
    model_path = os.path.join(
        os.path.expanduser("~"), ".cache", "mediapipe", "face_landmarker.task"
    )

    """Download the FaceLandmarker .task file if it is not already present."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[facial_parameters] Downloading FaceLandmarker model to:\n  {model_path}")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("[facial_parameters] Download complete.")
 
    h, w = img.shape[:2]
 
    # ── Build FaceLandmarker (Tasks API, replaces mp.solutions.face_mesh) ────
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
 
    # ── Convert BGR numpy array -> mediapipe.Image (RGB) ────────────────────
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
 
    # ── Run inference ────────────────────────────────────────────────────────
    with mp_vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)
 
    if not result.face_landmarks:
        return None  # no face detected
 
    # result.face_landmarks[0] is a list of NormalizedLandmark (x, y, z ∈ [0,1])
    lms = result.face_landmarks[0]
 
    # ── Face bounding box for spatial normalisation ──────────────────────────
    face_top_pt   = _lm_xy(lms, FACE_TOP,   w, h)
    face_bot_pt   = _lm_xy(lms, FACE_BOT,   w, h)
    face_left_pt  = _lm_xy(lms, FACE_LEFT,  w, h)
    face_right_pt = _lm_xy(lms, FACE_RIGHT, w, h)
 
    face_h = float(np.linalg.norm(face_bot_pt   - face_top_pt))  + 1e-6
    face_w = float(np.linalg.norm(face_right_pt - face_left_pt)) + 1e-6
 
    def norm_pt(pt: np.ndarray) -> np.ndarray:
        """Map pixel (x, y) into face-bounding-box-relative [0, 1] space."""
        return np.array([
            (pt[0] - face_left_pt[0]) / face_w,
            (pt[1] - face_top_pt[1])  / face_h,
        ], dtype=np.float32)
 
    # ── Left eye  [indices 0-4] ───────────────────────────────────────────────
    l_eye_pts    = [_lm_xy(lms, i, w, h) for i in LEFT_EYE]
    l_eye_center = norm_pt(np.mean(l_eye_pts, axis=0))
    l_eye_w      = float(np.linalg.norm(l_eye_pts[0] - l_eye_pts[3])) / face_w
    l_eye_h_val  = float(
        (np.linalg.norm(l_eye_pts[1] - l_eye_pts[5]) +
         np.linalg.norm(l_eye_pts[2] - l_eye_pts[4])) / 2.0
    ) / face_h
    l_ear = _eye_aspect_ratio(lms, LEFT_EYE, w, h)
 
    # ── Right eye  [5-9] ──────────────────────────────────────────────────────
    r_eye_pts    = [_lm_xy(lms, i, w, h) for i in RIGHT_EYE]
    r_eye_center = norm_pt(np.mean(r_eye_pts, axis=0))
    r_eye_w      = float(np.linalg.norm(r_eye_pts[0] - r_eye_pts[3])) / face_w
    r_eye_h_val  = float(
        (np.linalg.norm(r_eye_pts[1] - r_eye_pts[5]) +
         np.linalg.norm(r_eye_pts[2] - r_eye_pts[4])) / 2.0
    ) / face_h
    r_ear = _eye_aspect_ratio(lms, RIGHT_EYE, w, h)
 
    # ── Eyebrows  [10-15] ─────────────────────────────────────────────────────
    l_brow_pts    = [_lm_xy(lms, i, w, h) for i in LEFT_BROW]
    l_brow_center = norm_pt(np.mean(l_brow_pts, axis=0))
    l_brow_curv   = _brow_curvature(lms, LEFT_BROW,  w, h) / face_h
 
    r_brow_pts    = [_lm_xy(lms, i, w, h) for i in RIGHT_BROW]
    r_brow_center = norm_pt(np.mean(r_brow_pts, axis=0))
    r_brow_curv   = _brow_curvature(lms, RIGHT_BROW, w, h) / face_h
 
    # ── Nose  [16-18] ─────────────────────────────────────────────────────────
    nose_tip_px    = _lm_xy(lms, NOSE_TIP,    w, h)
    nose_bridge_px = _lm_xy(lms, NOSE_BRIDGE, w, h)
    nose_tip_norm  = norm_pt(nose_tip_px)
    nose_length    = float(np.linalg.norm(nose_tip_px - nose_bridge_px)) / face_h
 
    # ── Mouth  [19-22] ────────────────────────────────────────────────────────
    m_left       = _lm_xy(lms, MOUTH_LEFT,  w, h)
    m_right      = _lm_xy(lms, MOUTH_RIGHT, w, h)
    mouth_center = norm_pt((m_left + m_right) / 2.0)
    interocular  = float(np.linalg.norm(
        np.mean(l_eye_pts, axis=0) - np.mean(r_eye_pts, axis=0)
    ))
    mouth_width = float(np.linalg.norm(m_left - m_right)) / (interocular + 1e-6)
    mouth_curv  = _mouth_curvature(lms, w, h) / face_h
 
    # ── Interocular distance (normalised)  [23] ──────────────────────────────
    iod_norm = interocular / face_w
 
    # ── Assemble and return ──────────────────────────────────────────────────
    return np.array([
        # Left eye        [0-4]
        l_eye_center[0], l_eye_center[1],
        l_eye_w, l_eye_h_val,
        l_ear,
        # Right eye       [5-9]
        r_eye_center[0], r_eye_center[1],
        r_eye_w, r_eye_h_val,
        r_ear,
        # Left brow       [10-12]
        l_brow_center[0], l_brow_center[1],
        l_brow_curv,
        # Right brow      [13-15]
        r_brow_center[0], r_brow_center[1],
        r_brow_curv,
        # Nose            [16-18]
        nose_tip_norm[0], nose_tip_norm[1],
        nose_length,
        # Mouth           [19-22]
        mouth_center[0], mouth_center[1],
        mouth_width, mouth_curv,
        # Global          [23]
        iod_norm,
    ], dtype=np.float32)

# ----------------------------------------------------------------------
# 3. Generación del dataset (1000 imágenes con ruido)
# ----------------------------------------------------------------------

# Por cada directorio ( categoria )
# Por cada imagen extraer los parametros de ojos, etc usando opencv
# Definir esos parametros como x, y la salida (folder) como Y
def generar_dataset():
    X, y = [], []

    sizes = []
    xsizes = []
    for d in os.scandir(DIR_NAME):
        if d.is_dir():
            i = 0
            for f in os.scandir(d):
                if os.path.isfile(f):
                    i += 1
            sizes.append(i)
            print(i)

    num_data = min(sizes)
    print("papu")

    i = 0
    for d in os.scandir(DIR_NAME):
        if d.is_dir():
            j = 0
            for f in os.scandir(d):
                if j >= num_data:
                    break
                img = cv2.imread(f.path)
                x = extract_params(img)
                if x is None:
                    continue
                X.append(x)
                y.append(i)
                xsizes.append(len(x))
            i += 1
            print(j)
    for i in xsizes:
        print(i)

    return np.array(X), np.array(y)

X, y = generar_dataset()
print(f"X.shape = {X.shape},  y.shape = {y.shape}")

# ----------------------------------------------------------------------
# 4. Construcción del símplex regular (10 vértices en R^9)
#    Idea central del paper: targets equidistantes y simétricos
# ----------------------------------------------------------------------
def simplex_regular(m):
    """Construye los m vértices de un símplex regular en R^(m-1)."""
    T = np.zeros((m, m - 1))
    T[0, 0] = 1.0
    for i in range(1, m):
        T[i, 0] = -1.0 / (m - 1)
    for k in range(1, m - 1):
        T[k, k] = np.sqrt(1 - np.sum(T[k, :k]**2))
        for i in range(k + 1, m):
            T[i, k] = -T[k, k] / (m - k - 1)
    return T

T = simplex_regular(N_IDENTIDADES)   # (10, 9)
print(f"\nSímplex regular construido: {T.shape}")
print(f"Distancias por pares (deben ser iguales):")
print(f"  ||T0 - T1|| = {np.linalg.norm(T[0]-T[1]):.4f}")
print(f"  ||T3 - T7|| = {np.linalg.norm(T[3]-T[7]):.4f}")

# ----------------------------------------------------------------------
# 5. Entrenamiento Ridge multivariado contra los vértices del símplex
# ----------------------------------------------------------------------
Y = T[y]   # cada imagen recibe como label el vértice de su identidad

X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(
    X, Y, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nEntrenamiento: {X_tr.shape[0]} imgs | Prueba: {X_te.shape[0]} imgs")

ridge = Ridge(alpha=1.0)
ridge.fit(X_tr, Y_tr)

# ----------------------------------------------------------------------
# 6. Predicción: vecino más cercano al vértice del símplex
# ----------------------------------------------------------------------
def predecir(modelo, X, vertices):
    Y_hat = modelo.predict(X)
    # Distancia de cada predicción a cada vértice del símplex
    dists = np.linalg.norm(Y_hat[:, None, :] - vertices[None, :, :], axis=2)
    return np.argmin(dists, axis=1)

y_pred_tr = predecir(ridge, X_tr, T)
y_pred_te = predecir(ridge, X_te, T)

acc_tr = accuracy_score(y_tr, y_pred_tr)
acc_te = accuracy_score(y_te, y_pred_te)
print(f"\n>>> Accuracy entrenamiento: {acc_tr*100:.2f}%")
print(f">>> Accuracy prueba:        {acc_te*100:.2f}%")

# ----------------------------------------------------------------------
# 7. Visualización: una cara prototipo de cada identidad
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
dirs = [f.path for f in os.scandir(DIR_NAME) if f.is_dir()]
for i, ax in enumerate(axes.flat):
    imname = os.path.join(dirs[i], random.choice(os.listdir(dirs[i])))
    img = cv2.imread(imname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"Identidad {i}", fontsize=10)
    ax.axis("off")
plt.suptitle("Las 10 identidades prototípicas", fontweight="bold")
plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------
# 8. Visualización: variabilidad dentro de una identidad
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for ax in axes.flat:
    imname = os.path.join(dirs[i], random.choice(os.listdir(dirs[i])))
    img = cv2.imread(imname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap="gray_r"); ax.axis("off")
plt.suptitle("10 variantes ruidosas de la identidad 3", fontweight="bold")
plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------
# 9. Matriz de confusión
# ----------------------------------------------------------------------
cm = confusion_matrix(y_te, y_pred_te)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("Identidad predicha"); ax.set_ylabel("Identidad real")
ax.set_title(f"Matriz de confusión — Ridge (acc {acc_te*100:.1f}%)",
             fontweight="bold")
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=9)
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------
# 10. Visualización: aciertos y errores en el conjunto de prueba
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 6, figsize=(13, 5))
idxs = RNG.choice(range(10), 12)
for i,ax in enumerate(axes.flat):
    real = idxs[i]
    imname = os.path.join(dirs[real], random.choice(os.listdir(dirs[real])))
    img = cv2.imread(imname)
    params = extract_params(img)
    params = np.reshape(params, (1,-1))
    pred = predecir(ridge, params, T)
    color = "green" if real == pred else "red"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"Real {real} → Pred {pred}", color=color, fontsize=9)
    ax.axis("off")

plt.suptitle("Resultados sobre imágenes de prueba (verde=acierto, rojo=error)",
             fontweight="bold")
plt.tight_layout(); plt.show()

print("\n" + "="*60)
print("Concepto clave (paper): los 10 targets son vértices de un")
print("símplex regular en R^9 → puntos equidistantes y simétricos.")
print("Ridge mapea cada imagen cerca de su vértice y se clasifica")
print("por distancia mínima al vértice más cercano.")
print("="*60)
