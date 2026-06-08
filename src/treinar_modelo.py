import os
import random
from pathlib import Path

import tensorflow as tf

# Pasta base onde estao as vagas recortadas
DATA_DIR = Path(r"C:\Users\mp123\OneDrive\Documentos\treinamentoModeloArduino\imagensdasvvagas\recortes")
MODEL_OUT = Path(r"C:\Users\mp123\OneDrive\Documentos\treinamentoModeloArduino\modelo_vagas.keras")

# Ajustes principais
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42


def collect_images():
    # Espera estrutura: recortes/vagaX/cheia e recortes/vagaX/vazia
    items = []
    for vaga_dir in DATA_DIR.glob("vaga*"):
        if not vaga_dir.is_dir():
            continue
        for label_dir in (vaga_dir / "cheia", vaga_dir / "vazia"):
            if not label_dir.exists():
                continue
            label = 1 if label_dir.name == "cheia" else 0
            for img_path in label_dir.glob("*.jpg"):
                items.append((str(img_path), label))
    return items


def build_dataset(items):
    paths, labels = zip(*items)
    path_ds = tf.data.Dataset.from_tensor_slices(list(paths))
    label_ds = tf.data.Dataset.from_tensor_slices(list(labels))

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def main():
    items = collect_images()
    if not items:
        print("Nenhuma imagem encontrada em recortes/vaga*/(cheio|vazio).")
        return

    random.seed(SEED)
    random.shuffle(items)

    split = int(0.8 * len(items))
    train_items = items[:split]
    val_items = items[split:]

    train_ds = build_dataset(train_items)
    val_ds = build_dataset(val_items)

    train_ds = train_ds.shuffle(512, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.save(MODEL_OUT)
    print("Modelo salvo em:", MODEL_OUT)


if __name__ == "__main__":
    main()