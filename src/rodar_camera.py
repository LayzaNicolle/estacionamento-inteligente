import cv2
import numpy as np
import tensorflow as tf

# Ajuste para o caminho do modelo salvo
MODEL_PATH = r"C:\Users\mp123\OneDrive\Documentos\treinamentoModeloArduino\modelo_vagas.keras"

# Ajuste para o indice da camera (0, 1, 2...)
CAMERA_INDEX = 0

# Coordenadas no formato (x, y, largura, altura) - mesmas usadas no recorte
ROIS = [
    (71, 17, 104, 155),  # Vaga 1
    (197, 23, 117, 149),  # Vaga 2
    (320, 23, 113, 147),  # Vaga 3
]
IMG_SIZE = (160, 160)
THRESHOLD = 0.5  # >= 0.5 = cheia


def preprocess(roi_bgr: np.ndarray) -> np.ndarray:
    roi = cv2.resize(roi_bgr, IMG_SIZE)
    roi = roi.astype(np.float32) / 255.0
    return roi


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    # Tenta abrir com backends comuns do Windows
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if cap.isOpened():
            break
        cap.release()
        cap = None

    if cap is None or not cap.isOpened():
        print("Nao foi possivel abrir a camera.")
        print("Tente trocar CAMERA_INDEX (0, 1, 2...) e feche apps que usam a camera.")
        return

    print("Pressione Q para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler a camera. Tente outro CAMERA_INDEX ou feche apps que usam a camera.")
            break

        preds = []
        frame_h, frame_w = frame.shape[:2]
        for (x, y, w, h) in ROIS:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_w, x + w)
            y2 = min(frame_h, y + h)
            if x2 <= x1 or y2 <= y1:
                preds.append(0)
                continue

            roi = frame[y1:y2, x1:x2]
            inp = preprocess(roi)
            inp = np.expand_dims(inp, axis=0)
            prob = float(model.predict(inp, verbose=0)[0][0])
            occupied = 1 if prob >= THRESHOLD else 0
            preds.append(occupied)

        # Vetor de saida: vaga1=pos0, vaga2=pos1, vaga3=pos2
        print(preds)

        # Opcional: mostrar a imagem com retangulos
        for (x, y, w, h) in ROIS:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()