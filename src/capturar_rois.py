import os
import cv2

# Para usar a camera ao vivo para capturar um frame de referencia.
CAMERA_INDEX = 0
OUTPUT_TXT = r"C:\Users\mp123\OneDrive\Documentos\treinamentoModeloArduino\rois.txt"

rois = []


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Nao foi possivel abrir a camera. Tente trocar CAMERA_INDEX.")
        return

    ret, img = cap.read()
    cap.release()
    if not ret or img is None:
        print("Falha ao capturar frame da camera.")
        return

    # Selecao interativa de ROIs
    print("Selecione 3 ROIs (vaga 1, 2, 3). Pressione ENTER para confirmar cada uma.")
    print("Pressione ESC quando terminar a selecao.")

    temp = img.copy()
    rois_selected = cv2.selectROIs("Selecionar ROIs", temp, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if rois_selected is None or len(rois_selected) == 0:
        print("Nenhuma ROI selecionada.")
        return

    # Salvar no formato (x, y, w, h)
    lines = []
    for idx, (x, y, w, h) in enumerate(rois_selected, start=1):
        lines.append(f"({int(x)}, {int(y)}, {int(w)}, {int(h)})  # Vaga {idx}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("ROIS = [\n")
        for line in lines:
            f.write(f"    {line},\n")
        f.write("]\n")

    print("ROIs salvas em:", OUTPUT_TXT)
    print("Copie a lista ROIS para o recortar_imagens.py")


if __name__ == "__main__":
    main()
