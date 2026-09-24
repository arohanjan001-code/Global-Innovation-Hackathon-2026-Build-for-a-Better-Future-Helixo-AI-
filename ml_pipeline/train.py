"""
Helixo AI — High-Accuracy Helmet Detection Model v2
Target: 95%+ validation accuracy
Approach: Transfer learning with MobileNetV2 + partial fine-tuning + mixup
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import random

# ─── Step 1: Augment no_helmet images ───
print("=" * 60)
print("  Step 1: Balancing dataset")
print("=" * 60)

src_dir = "dataset/no_helmet"
valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

# Clean previous augmented
for f in os.listdir(src_dir):
    if f.startswith('aug_') and f.lower().endswith(valid_ext):
        os.remove(os.path.join(src_dir, f))

originals = []
for f in sorted(os.listdir(src_dir)):
    if f.startswith('.') or os.path.isdir(os.path.join(src_dir, f)):
        continue
    if not f.lower().endswith(valid_ext):
        continue
    img = cv2.imread(os.path.join(src_dir, f))
    if img is not None:
        originals.append(f)
print(f"  No-helmet originals: {len(originals)}")

helmet_dir = "dataset/helmet"
helmet_count = 0
for f in os.listdir(helmet_dir):
    if f.startswith('.') or os.path.isdir(os.path.join(helmet_dir, f)):
        continue
    if cv2.imread(os.path.join(helmet_dir, f)) is not None:
        helmet_count += 1
print(f"  Helmet originals: {helmet_count}")

# More aggressive augmentation with broader variety
augments_needed = helmet_count - len(originals)
if augments_needed > 0:
    print(f"  Generating {augments_needed} augmented images (diverse transforms)...")
    count, idx = 0, 0
    while count < augments_needed:
        for fname in originals:
            if count >= augments_needed:
                break
            img = cv2.imread(os.path.join(src_dir, fname))
            if img is None:
                continue
            aug = img.copy()

            # Random horizontal flip
            if random.random() > 0.5:
                aug = cv2.flip(aug, 1)

            # Random rotation (-35 to +35)
            angle = random.uniform(-35, 35)
            h, w = aug.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Random crop (more aggressive)
            cf = random.uniform(0.65, 0.95)
            ch, cw = int(h * cf), int(w * cf)
            y, x = random.randint(0, h - ch), random.randint(0, w - cw)
            aug = aug[y:y+ch, x:x+cw]
            aug = cv2.resize(aug, (w, h))

            # Random brightness + contrast
            alpha = random.uniform(0.7, 1.3)  # contrast
            beta = random.randint(-30, 30)      # brightness
            aug = np.clip(aug * alpha + beta, 0, 255).astype(np.uint8)

            # Random color jitter (hue/saturation)
            if random.random() > 0.5:
                hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.int16)
                hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-15, 15)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
                aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Random Gaussian blur
            if random.random() > 0.7:
                k = random.choice([3, 5])
                aug = cv2.GaussianBlur(aug, (k, k), 0)

            # Random noise
            if random.random() > 0.6:
                noise = np.random.normal(0, random.randint(5, 20), aug.shape).astype(np.int16)
                aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(src_dir, f"aug_{idx:04d}.jpg"), aug,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            idx += 1
            count += 1
    print(f"  ✅ Generated {count} augmented images")

# ─── Step 2: Load images ───
print("\n" + "=" * 60)
print("  Step 2: Loading all images")
print("=" * 60)

IMG_SIZE = 160  # Higher resolution for more detail
BATCH_SIZE = 16

images, labels = [], []


def load_folder(folder, label):
    loaded = 0
    for f in sorted(os.listdir(folder)):
        if f.startswith('.') or os.path.isdir(os.path.join(folder, f)):
            continue
        if not f.lower().endswith(valid_ext):
            continue
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)
        loaded += 1
    return loaded


n_helmet = load_folder("dataset/helmet", 0)
n_no_helmet = load_folder("dataset/no_helmet", 1)
print(f"  Loaded: {n_helmet} helmet, {n_no_helmet} no_helmet")

X = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.float32)
del images, labels

# Shuffle
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

# Split 80/20
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

# Class weights
total = n_helmet + n_no_helmet
class_weights = {0: total / (2 * n_helmet), 1: total / (2 * n_no_helmet)}
print(f"  Class weights: {class_weights}")

# ─── Step 3: Build Model ───
print("\n" + "=" * 60)
print("  Step 3: Building MobileNetV2 model")
print("=" * 60)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.5,  # Slightly bigger for more capacity
    include_top=False,
    weights='imagenet'
)

# Build model with functional API for more control
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Lambda(lambda img: preprocess_input(img))(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

total_layers = len(base_model.layers)
print(f"  MobileNetV2 layers: {total_layers}")
print(f"  Total params: {model.count_params():,}")

# Data augmentation
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])


# Mixup augmentation (blends pairs of images for novel samples)
def mixup_batch(images, labels, alpha=0.2):
    """Mixup: blend pairs of images/labels for regularization."""
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    lam = tf.random.uniform([], 0, alpha)
    images = lam * shuffled_images + (1 - lam) * images
    labels = lam * shuffled_labels + (1 - lam) * labels
    return images, labels


def prepare_train_batch(images, labels):
    images = data_aug(images, training=True)
    # Apply mixup 50% of the time
    if tf.random.uniform([]) > 0.5:
        images, labels = mixup_batch(images, labels)
    return images, labels


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(2000).batch(BATCH_SIZE).map(
    prepare_train_batch, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ─── Step 4: Phase 1 — Train head only ───
print("\n" + "=" * 60)
print("  Step 4: Phase 1 — Training head (base frozen)")
print("=" * 60)

base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

h1 = model.fit(
    train_ds, validation_data=val_ds, epochs=15,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1
        )
    ]
)
phase1_acc = max(h1.history['val_accuracy'])
print(f"\n  ✅ Phase 1 best val_accuracy: {phase1_acc:.4f}")

# ─── Step 5: Phase 2 — Fine-tune last 30 layers ───
print("\n" + "=" * 60)
print("  Step 5: Phase 2 — Fine-tuning last 30 layers")
print("=" * 60)

# Unfreeze only the last 30 layers of base_model
base_model.trainable = True
fine_tune_from = max(0, total_layers - 30)
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"  Fine-tuning {trainable_count}/{total_layers} base layers")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

h2 = model.fit(
    train_ds, validation_data=val_ds, epochs=40,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'helmet_model_best.h5', monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-7, verbose=1
        )
    ]
)

# ─── Step 6: Phase 3 — Full unfreeze, very low LR ───
phase2_acc = max(h2.history['val_accuracy'])
print(f"\n  ✅ Phase 2 best val_accuracy: {phase2_acc:.4f}")

if phase2_acc < 0.95:
    print("\n" + "=" * 60)
    print("  Step 6: Phase 3 — Full fine-tune (all layers, ultra-low LR)")
    print("=" * 60)

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-6),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    h3 = model.fit(
        train_ds, validation_data=val_ds, epochs=30,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=8,
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'helmet_model_best.h5', monitor='val_accuracy',
                save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3,
                min_lr=1e-8, verbose=1
            )
        ]
    )
    phase3_acc = max(h3.history['val_accuracy'])
    print(f"\n  ✅ Phase 3 best val_accuracy: {phase3_acc:.4f}")
    final_acc = phase3_acc
else:
    final_acc = phase2_acc

# ─── Results ───
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)

# Training accuracy (last epoch of each phase)
phase1_train_acc = max(h1.history['accuracy'])
phase2_train_acc = max(h2.history['accuracy'])

print(f"\n  {'Phase':<25} {'Train Acc':>12} {'Val Acc':>12}")
print(f"  {'─'*25} {'─'*12} {'─'*12}")
print(f"  {'Phase 1 (head only)':<25} {phase1_train_acc*100:>10.2f}%  {phase1_acc*100:>10.2f}%")
print(f"  {'Phase 2 (partial tune)':<25} {phase2_train_acc*100:>10.2f}%  {phase2_acc*100:>10.2f}%")

if phase2_acc < 0.95:
    phase3_train_acc = max(h3.history['accuracy'])
    print(f"  {'Phase 3 (full tune)':<25} {phase3_train_acc*100:>10.2f}%  {final_acc*100:>10.2f}%")
    final_train_acc = phase3_train_acc
else:
    final_train_acc = phase2_train_acc

print(f"\n  {'FINAL':<25} {final_train_acc*100:>10.2f}%  {final_acc*100:>10.2f}%")

# Overfitting check
overfit_gap = (final_train_acc - final_acc) * 100
if overfit_gap > 10:
    print(f"\n  ⚠️  Overfitting detected! Train-Val gap: {overfit_gap:.1f}%")
    print(f"      Consider adding more training data or increasing regularization.")
elif overfit_gap > 5:
    print(f"\n  📊 Slight overfitting. Train-Val gap: {overfit_gap:.1f}%")
else:
    print(f"\n  ✅ No significant overfitting. Train-Val gap: {overfit_gap:.1f}%")

if final_acc >= 0.95:
    print("\n  🎉 TARGET ACHIEVED! 95%+ accuracy!")
elif final_acc >= 0.93:
    print(f"\n  📊 Very close at {final_acc*100:.1f}%! More unique images would push past 95%.")
else:
    print(f"\n  ⚠️  {final_acc*100:.1f}% — dataset diversity is the bottleneck.")

model.save('helmet_model.h5')
print("  ✅ Models saved: helmet_model_best.h5, helmet_model.h5")