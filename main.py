from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

app = FastAPI()


# ------------------------
# 🧠 ANALYSIS FUNCTION
# ------------------------
def analyze_card(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    card = img[y:y+h, x:x+w]

    gray_card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    gray_card = cv2.equalizeHist(gray_card)

    # ------------------------
    # CENTERING
    # ------------------------
    edges_card = cv2.Canny(gray_card, 100, 200)

    col_sum = np.sum(edges_card, axis=0)
    row_sum = np.sum(edges_card, axis=1)

    left_edge = np.argmax(col_sum[:w//3])
    right_edge = np.argmax(col_sum[2*w//3:]) + 2*w//3

    top_edge = np.argmax(row_sum[:h//3])
    bottom_edge = np.argmax(row_sum[2*h//3:]) + 2*h//3

    left = left_edge
    right = w - right_edge
    top = top_edge
    bottom = h - bottom_edge

    horiz = max(left, right) / (left + right + 1e-5)
    vert = max(top, bottom) / (top + bottom + 1e-5)

    centering = round((horiz + vert) / 2, 2)

    # ------------------------
    # CORNERS
    # ------------------------
    corner_size = int(min(w, h) * 0.08)

    corners = [
        gray_card[0:corner_size, 0:corner_size],
        gray_card[0:corner_size, w-corner_size:w],
        gray_card[h-corner_size:h, 0:corner_size],
        gray_card[h-corner_size:h, w-corner_size:w]
    ]

    corner_damage = 0

    for c in corners:
        _, thresh = cv2.threshold(c, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(thresh == 255) / (c.size + 1e-5)

        if white_ratio > 0.01:
            corner_damage += white_ratio

    corner_damage = round(corner_damage, 3)

    # ------------------------
    # EDGES
    # ------------------------
    border = int(min(w, h) * 0.05)

    regions = [
        gray_card[:, :border],
        gray_card[:, w-border:],
        gray_card[:border, :],
        gray_card[h-border:, :]
    ]

    edge_damage = 0

    for r in regions:
        e = cv2.Canny(r, 50, 150)
        edge_damage += np.sum(e > 0) / (r.size + 1e-5)

    edge_damage = round(edge_damage / 4, 3)

    # ------------------------
    # SURFACE
    # ------------------------
    blur2 = cv2.GaussianBlur(gray_card, (9, 9), 0)
    diff = cv2.absdiff(gray_card, blur2)

    noise = round(np.sum(diff > 25) / (diff.size + 1e-5), 3)

    # ------------------------
    # SCORE POR PARTE
    # ------------------------
    score = 10
    score -= abs(centering - 0.5) * 5
    score -= corner_damage * 6
    score -= edge_damage * 5
    score -= noise * 3

    score = round(max(0, min(10, score)), 1)

    return {
        "centering": centering,
        "corners": corner_damage,
        "edges": edge_damage,
        "surface": noise,
        "score": score
    }


# ------------------------
# 🚀 ENDPOINT FINAL
# ------------------------
@app.post("/grade")
async def grade(
    front: UploadFile = File(...),
    back: UploadFile = File(...)
):
    try:
        front_img = cv2.imdecode(np.frombuffer(await front.read(), np.uint8), cv2.IMREAD_COLOR)
        back_img = cv2.imdecode(np.frombuffer(await back.read(), np.uint8), cv2.IMREAD_COLOR)

        front_data = analyze_card(front_img)
        back_data = analyze_card(back_img)

        if front_data is None or back_data is None:
            return {"error": "No se pudo analizar una de las caras"}

        # ------------------------
        # 🧠 FINAL SCORE
        # ------------------------
        base_score = front_data["score"] * 0.6 + back_data["score"] * 0.4

        if abs(front_data["score"] - back_data["score"]) > 2:
            base_score -= 0.5

        final_score = round(max(0, min(10, base_score)), 1)

        return {
            "front": front_data,
            "back": back_data,
            "final_score": final_score
        }

    except Exception as e:
        return {"error": str(e)}