from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API funcionando"}

# ===============================
# FUNCIONES IA
# ===============================

def detect_card(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_rect = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                max_area = area
                best_rect = approx

    return best_rect


def crop_card(img, rect):
    pts = rect.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warp


def detect_corner_damage(card):
    h, w = card.shape[:2]
    size = int(min(w, h) * 0.1)

    corners = [
        card[0:size, 0:size],
        card[0:size, w-size:w],
        card[h-size:h, 0:size],
        card[h-size:h, w-size:w]
    ]

    damage_score = 0

    for corner in corners:
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        white_pixels = np.sum(thresh == 255)
        total_pixels = thresh.size

        ratio = white_pixels / total_pixels

        if ratio > 0.05:
            damage_score += 2

    return damage_score


def calculate_centering(card):
    h, w = card.shape[:2]

    left = card[:, :int(w*0.1)]
    right = card[:, int(w*0.9):]
    top = card[:int(h*0.1), :]
    bottom = card[int(h*0.9):, :]

    left_mean = np.mean(left)
    right_mean = np.mean(right)
    top_mean = np.mean(top)
    bottom_mean = np.mean(bottom)

    horizontal = f"{int(left_mean)} / {int(right_mean)}"
    vertical = f"{int(top_mean)} / {int(bottom_mean)}"

    return horizontal, vertical


def calculate_score(damage, ratio):
    score = 10.0

    score -= damage * 0.5

    if ratio < 0.55 or ratio > 0.75:
        score -= 1.5

    return round(max(score, 0), 1)


# ===============================
# ENDPOINT
# ===============================

@app.post("/grade")
async def grade(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Imagen inválida"}

        rect = detect_card(img)

        if rect is None:
            return {
                "card_detected": False,
                "message": "No se detecta carta"
            }

        card = crop_card(img, rect)

        h, w = card.shape[:2]
        ratio = w / h

        damage = detect_corner_damage(card)
        cent_h, cent_v = calculate_centering(card)

        score = calculate_score(damage, ratio)

        return {
            "card_detected": True,
            "ratio": round(ratio, 2),
            "corner_damage_score": int(damage),
            "centering_horizontal": cent_h,
            "centering_vertical": cent_v,
            "final_score": score,
            "message": "Grading completado"
        }

    except Exception as e:
        return {"error": str(e)}