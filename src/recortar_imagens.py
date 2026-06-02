import os
import cv2

INPUT_DIR = r"C:\Users\mp123\OneDrive\Documentos\treinamentoModeloArduino\imagensdasvvagas"
OUTPUT_DIR = os.path.join(INPUT_DIR, "recortes")
EXTENSIONS = (".jpg", ".jpeg", ".png")

# Coordenadas no formato (x, y, largura, altura)
ROIS = [
    (377, 48, 256, 332),  # Vaga 1
    (660, 49, 274, 337),  # Vaga 2
    (941, 52, 268, 331),  # Vaga 3
]


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for idx in range(len(ROIS)):
        os.makedirs(os.path.join(OUTPUT_DIR, f"vaga{idx + 1}"), exist_ok=True)


def is_image_file(name: str) -> bool:
    return name.lower().endswith(EXTENSIONS)


def main():
    ensure_dirs()

    images = [f for f in os.listdir(INPUT_DIR) if is_image_file(f)]
    if not images:
        print("Nenhuma imagem encontrada.")
        return

    for name in images:
        src_path = os.path.join(INPUT_DIR, name)
        img = cv2.imread(src_path)
        if img is None:
            print(f"Falha ao ler: {name}")
            continue

        h, w = img.shape[:2]
        for idx, (x, y, roi_w, roi_h) in enumerate(ROIS):
            x2 = x + roi_w
            y2 = y + roi_h

            # Validacao simples para evitar cortes fora da imagem
            if x < 0 or y < 0 or x2 > w or y2 > h:
                print(f"ROI fora da imagem em {name}: vaga {idx + 1}")
                continue

            crop = img[y:y2, x:x2]
            out_dir = os.path.join(OUTPUT_DIR, f"vaga{idx + 1}")
            out_path = os.path.join(out_dir, name)
            cv2.imwrite(out_path, crop)

    print("Recortes concluidos.")


if __name__ == "__main__":
    main()
