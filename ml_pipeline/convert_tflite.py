"""
Helixo AI — TFLite Conversion for ESP32-S3
Converts trained helmet_model_best.h5 → INT8 quantized TFLite
Generates:
  1. helmet_model_esp32.tflite  (for deployment)
  2. helmet_model_esp32.h       (C header for direct embedding)

Uses SavedModel export to avoid MLIR converter bugs.
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import tempfile
import shutil

# ─── Configuration ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 160
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp')

print("=" * 60)
print("  🏍️  HELIXO AI — TFLite Conversion for ESP32-S3")
print("=" * 60)

# ─── Step 1: Rebuild and load trained model ───
print("\n📦 Step 1: Loading trained model...")

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.5,
    include_top=False,
    weights='imagenet'
)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Lambda(
    lambda img: preprocess_input(img),
    output_shape=(IMG_SIZE, IMG_SIZE, 3)
)(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# Load best weights
weights_path = os.path.join(SCRIPT_DIR, "helmet_model_best.h5")
if not os.path.exists(weights_path):
    weights_path = os.path.join(SCRIPT_DIR, "helmet_model.h5")

model.load_weights(weights_path)
print(f"  ✅ Loaded: {os.path.basename(weights_path)}")
print(f"  Parameters: {model.count_params():,}")

# ─── Step 2: Load representative dataset for quantization ───
print("\n📊 Step 2: Loading representative dataset for quantization...")

representative_images = []
for class_name in ["helmet", "no_helmet"]:
    class_dir = os.path.join(DATASET_DIR, class_name)
    count = 0
    for f in sorted(os.listdir(class_dir)):
        if f.startswith('.') or not f.lower().endswith(VALID_EXT):
            continue
        img = cv2.imread(os.path.join(class_dir, f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        representative_images.append(img.astype(np.float32))
        count += 1
        if count >= 100:  # 100 per class = 200 total
            break
    print(f"  Loaded {count} {class_name} images")

representative_images = np.array(representative_images)
print(f"  Total representative samples: {len(representative_images)}")

# Shuffle for better quantization calibration
np.random.seed(42)
np.random.shuffle(representative_images)


# ─── Step 3: Save as SavedModel first, then convert ───
print("\n⚙️  Step 3: Converting to TFLite...")

# Save as SavedModel to avoid MLIR bugs with direct Keras conversion
saved_model_dir = os.path.join(SCRIPT_DIR, "_saved_model_temp")
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

print("  Exporting SavedModel...")
try:
    # Keras 3 method
    model.export(saved_model_dir)
except (AttributeError, TypeError):
    # Keras 2 fallback
    tf.saved_model.save(model, saved_model_dir)
print("  ✅ SavedModel exported")

def representative_data_gen():
    """Yields representative data for INT8 calibration."""
    for i in range(min(200, len(representative_images))):
        sample = representative_images[i:i+1]
        yield [sample]

# Try multiple conversion strategies
tflite_model = None
conversion_type = ""

# Strategy 1: Full INT8 quantization
print("\n  Trying full INT8 quantization...")
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    conversion_type = "INT8 (Full Quantization)"
    print(f"  ✅ Full INT8 conversion successful!")
except Exception as e:
    print(f"  ⚠️  Full INT8 failed: {str(e)[:100]}")

# Strategy 2: Dynamic range quantization with INT8 I/O
if tflite_model is None:
    print("\n  Trying INT8 with fallback ops...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        conversion_type = "UINT8 (With Fallback)"
        print(f"  ✅ UINT8 conversion successful!")
    except Exception as e:
        print(f"  ⚠️  UINT8 failed: {str(e)[:100]}")

# Strategy 3: Dynamic range quantization (float I/O, quantized weights)
if tflite_model is None:
    print("\n  Trying dynamic range quantization...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        conversion_type = "Dynamic Range (Float I/O, Quantized Weights)"
        print(f"  ✅ Dynamic range conversion successful!")
    except Exception as e:
        print(f"  ⚠️  Dynamic range failed: {str(e)[:100]}")

# Strategy 4: Float16 quantization
if tflite_model is None:
    print("\n  Trying Float16 quantization...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        conversion_type = "Float16"
        print(f"  ✅ Float16 conversion successful!")
    except Exception as e:
        print(f"  ❌ Float16 failed: {str(e)[:100]}")

# Strategy 5: No quantization (baseline)
if tflite_model is None:
    print("\n  Trying unquantized conversion...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        conversion_type = "Unquantized (Float32)"
        print(f"  ✅ Unquantized conversion successful!")
    except Exception as e:
        print(f"  ❌ All conversions failed: {e}")
        # Cleanup
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        exit(1)

# Cleanup temp SavedModel
shutil.rmtree(saved_model_dir, ignore_errors=True)

# Save TFLite model
tflite_path = os.path.join(SCRIPT_DIR, "helmet_model_esp32.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

tflite_size = os.path.getsize(tflite_path)
h5_size = os.path.getsize(weights_path)
print(f"\n  Quantization type: {conversion_type}")
print(f"\n  📁 Model sizes:")
print(f"     Original (.h5):     {h5_size / 1024:.1f} KB ({h5_size / 1024 / 1024:.2f} MB)")
print(f"     TFLite:             {tflite_size / 1024:.1f} KB ({tflite_size / 1024 / 1024:.2f} MB)")
print(f"     Compression:        {h5_size / tflite_size:.1f}x smaller")


# ─── Step 4: Validate TFLite model accuracy ───
print("\n🧪 Step 4: Validating TFLite model accuracy...")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dtype = input_details[0]['dtype']
input_shape = input_details[0]['shape']
output_dtype = output_details[0]['dtype']

# Get quantization params if available
input_quant = input_details[0].get('quantization_parameters', {})
output_quant = output_details[0].get('quantization_parameters', {})

input_scale = input_quant.get('scales', [1.0])
input_zp = input_quant.get('zero_points', [0])
output_scale = output_quant.get('scales', [1.0])
output_zp = output_quant.get('zero_points', [0])

if len(input_scale) > 0:
    input_scale = input_scale[0]
else:
    input_scale = 1.0
if len(input_zp) > 0:
    input_zp = input_zp[0]
else:
    input_zp = 0
if len(output_scale) > 0:
    output_scale = output_scale[0]
else:
    output_scale = 1.0
if len(output_zp) > 0:
    output_zp = output_zp[0]
else:
    output_zp = 0

print(f"  Input:  shape={input_shape}, dtype={input_dtype}")
print(f"  Output: dtype={output_dtype}")
print(f"  Input quantization:  scale={input_scale}, zero_point={input_zp}")
print(f"  Output quantization: scale={output_scale}, zero_point={output_zp}")


def run_tflite_prediction(img_float):
    """Run prediction through TFLite interpreter."""
    if input_dtype == np.float32:
        img_input = img_float
    elif input_dtype == np.int8:
        img_input = (img_float / input_scale + input_zp).astype(np.int8)
    elif input_dtype == np.uint8:
        img_input = (img_float / input_scale + input_zp).astype(np.uint8)
    else:
        img_input = img_float

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img_input, 0))
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if output_dtype in [np.int8, np.uint8]:
        pred = (float(raw_output) - output_zp) * output_scale
    else:
        pred = float(raw_output)

    return pred


# Test on dataset
correct_keras = 0
correct_tflite = 0
total = 0

for class_name, label in [("helmet", 0), ("no_helmet", 1)]:
    class_dir = os.path.join(DATASET_DIR, class_name)
    tested = 0
    for f in sorted(os.listdir(class_dir)):
        if f.startswith('.') or not f.lower().endswith(VALID_EXT):
            continue
        img = cv2.imread(os.path.join(class_dir, f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_float = img.astype(np.float32)

        # Keras prediction
        keras_pred = model.predict(np.expand_dims(img_float, 0), verbose=0)[0][0]
        keras_label = 1 if keras_pred > 0.5 else 0

        # TFLite prediction
        tflite_pred = run_tflite_prediction(img_float)
        tflite_label = 1 if tflite_pred > 0.5 else 0

        if keras_label == label:
            correct_keras += 1
        if tflite_label == label:
            correct_tflite += 1
        total += 1
        tested += 1

        if tested >= 50:  # Test 50 per class for speed
            break

keras_acc = correct_keras / total * 100 if total > 0 else 0
tflite_acc = correct_tflite / total * 100 if total > 0 else 0

print(f"\n  Tested {total} images (50 per class)")
print(f"  Keras accuracy:     {keras_acc:.1f}%")
print(f"  TFLite accuracy:    {tflite_acc:.1f}%")
print(f"  Accuracy drop:      {keras_acc - tflite_acc:.1f}%")


# ─── Step 5: Generate C header file ───
print("\n📝 Step 5: Generating C header file for ESP32-S3...")

header_path = os.path.join(SCRIPT_DIR, "helmet_model_esp32.h")
with open(header_path, 'w') as f:
    f.write("// Helixo AI — Helmet Detection Model for ESP32-S3\n")
    f.write("// Auto-generated — DO NOT EDIT\n")
    f.write(f"// Model size: {tflite_size} bytes ({tflite_size / 1024:.1f} KB)\n")
    f.write(f"// Quantization: {conversion_type}\n")
    f.write(f"// Input: {IMG_SIZE}x{IMG_SIZE}x3 {input_dtype.__name__}\n")
    f.write(f"// Output: 1 neuron (sigmoid) — 0=helmet, 1=no_helmet\n")
    f.write(f"// Keras accuracy: {keras_acc:.1f}% | TFLite accuracy: {tflite_acc:.1f}%\n")
    f.write("\n")
    f.write("#ifndef HELMET_MODEL_ESP32_H\n")
    f.write("#define HELMET_MODEL_ESP32_H\n\n")
    f.write(f"const unsigned int helmet_model_len = {tflite_size};\n")
    f.write(f"alignas(16) const unsigned char helmet_model[] = {{\n")

    # Write model bytes as hex
    for i, byte in enumerate(tflite_model):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{byte:02x}")
        if i < len(tflite_model) - 1:
            f.write(", ")
        if (i + 1) % 12 == 0:
            f.write("\n")

    f.write("\n};\n\n")
    f.write("#endif  // HELMET_MODEL_ESP32_H\n")

header_size = os.path.getsize(header_path)
print(f"  ✅ Generated: helmet_model_esp32.h ({header_size / 1024:.1f} KB)")


# ─── Step 6: Test with test1 images ───
print("\n🏍️  Step 6: Testing TFLite model on test1/ (your bike photos)...")

test1_dir = os.path.join(SCRIPT_DIR, "test1")
if os.path.exists(test1_dir):
    print(f"\n  {'#':<3} {'Image':<28} {'Keras':>8} {'TFLite':>8} {'Match'}")
    print("  " + "-" * 55)

    test_correct = 0
    test_total = 0
    for i, f in enumerate(sorted(os.listdir(test1_dir)), 1):
        if f.startswith('.') or not f.lower().endswith(VALID_EXT):
            continue
        img = cv2.imread(os.path.join(test1_dir, f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

        # Keras
        k_pred = model.predict(np.expand_dims(img, 0), verbose=0)[0][0]
        k_conf = (1 - k_pred) * 100 if k_pred < 0.5 else k_pred * 100
        k_label = "HELMET" if k_pred < 0.5 else "NO HELMET"

        # TFLite
        t_pred = run_tflite_prediction(img)
        t_conf = (1 - t_pred) * 100 if t_pred < 0.5 else t_pred * 100
        t_label = "HELMET" if t_pred < 0.5 else "NO HELMET"

        match = "✅" if k_label == t_label else "❌"
        if k_label == t_label:
            test_correct += 1
        test_total += 1

        print(f"  {i:<3} {f:<28} {k_conf:>6.1f}%  {t_conf:>6.1f}%  {match}")

    if test_total > 0:
        print(f"\n  Match rate: {test_correct}/{test_total} ({test_correct/test_total*100:.0f}%)")
else:
    print("  ⚠️  test1/ directory not found, skipping")


# ─── Final Summary ───
print("\n" + "=" * 60)
print("  📊 CONVERSION SUMMARY")
print("=" * 60)
print(f"""
  Quantization:      {conversion_type}
  Original model:    {h5_size / 1024 / 1024:.2f} MB  ({os.path.basename(weights_path)})
  TFLite model:      {tflite_size / 1024:.1f} KB   (helmet_model_esp32.tflite)
  C Header:          {header_size / 1024:.1f} KB   (helmet_model_esp32.h)
  Compression:       {h5_size / tflite_size:.1f}x smaller

  Keras accuracy:    {keras_acc:.1f}%
  TFLite accuracy:   {tflite_acc:.1f}%
  Accuracy drop:     {keras_acc - tflite_acc:.1f}%

  ESP32-S3 Requirements:
    Flash:  {tflite_size / 1024:.0f} KB (model) + ~50 KB (TF Micro runtime)
    RAM:    ~{IMG_SIZE * IMG_SIZE * 3 / 1024:.0f} KB (input tensor) + ~200 KB (arena)
    Speed:  ~5-10 FPS estimated

  Files generated:
    ✅ helmet_model_esp32.tflite  → Flash to ESP32-S3
    ✅ helmet_model_esp32.h       → #include in Arduino/ESP-IDF code
""")

print("  🚀 Ready for ESP32-S3 deployment!")
print("=" * 60)
