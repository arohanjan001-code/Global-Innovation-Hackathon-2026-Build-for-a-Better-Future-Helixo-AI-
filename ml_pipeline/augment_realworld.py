"""
Helixo AI — Real-World Augmentation Engine
Generates 1000+ realistic training images with:
  - Night/dark lighting
  - Rain/fog/haze
  - Motion blur (bike riding)
  - Face mask simulation
  - Goggles/sunglasses overlay
  - Golden hour/sunset tint
  - Heavy shadows
  - Glare/overexposure
  - Image compositing (blending two images)
  - Random occlusion
"""

import cv2
import numpy as np
import os
import random

# ─── Configuration ───
DATASET_DIR = "dataset"
HELMET_DIR = os.path.join(DATASET_DIR, "helmet")
NO_HELMET_DIR = os.path.join(DATASET_DIR, "no_helmet")
TARGET_PER_CLASS = 550  # 550 + 550 = 1100+ total
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp')

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════
#  AUGMENTATION FUNCTIONS — Real World Simulations
# ═══════════════════════════════════════════════════

def night_mode(img):
    """Simulate night/dark environment — very dark with blue tint."""
    dark = img.astype(np.float32)
    # Darken significantly
    dark *= random.uniform(0.15, 0.35)
    # Add blue tint (night sky reflection)
    dark[:, :, 0] += random.randint(10, 30)  # Blue channel boost
    dark[:, :, 1] += random.randint(0, 10)    # Slight green
    return np.clip(dark, 0, 255).astype(np.uint8)


def dark_lowlight(img):
    """Simulate very low light / poorly lit area."""
    # Gamma correction for darkness
    gamma = random.uniform(2.0, 3.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    dark = cv2.LUT(img, table)
    # Add noise (low light = more noise)
    noise = np.random.normal(0, random.randint(15, 35), dark.shape).astype(np.int16)
    dark = np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return dark


def rain_effect(img):
    """Simulate rain with streaks and slight blur."""
    rain = img.copy()
    h, w = rain.shape[:2]
    # Add rain streaks
    num_drops = random.randint(100, 300)
    for _ in range(num_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(10, 30)
        thickness = random.choice([1, 1, 1, 2])
        # Rain drops are slightly diagonal
        x_end = x + random.randint(-3, 3)
        y_end = min(y + length, h - 1)
        cv2.line(rain, (x, y), (x_end, y_end), (200, 200, 220), thickness)
    # Slight blur for wet windshield effect
    rain = cv2.GaussianBlur(rain, (3, 3), 0)
    # Slightly darken (overcast)
    rain = (rain.astype(np.float32) * random.uniform(0.7, 0.85)).astype(np.uint8)
    return rain


def fog_haze(img):
    """Simulate foggy/hazy conditions."""
    fog = img.astype(np.float32)
    h, w = fog.shape[:2]
    # Create fog layer (white overlay with gradient)
    fog_intensity = random.uniform(0.3, 0.6)
    # Bottom has more fog
    for y_row in range(h):
        row_fog = fog_intensity * (0.5 + 0.5 * (y_row / h))
        fog[y_row] = fog[y_row] * (1 - row_fog) + 255 * row_fog
    # Blur for fog effect
    k = random.choice([5, 7, 9])
    fog = cv2.GaussianBlur(fog.astype(np.uint8), (k, k), 0)
    return fog.astype(np.uint8)


def motion_blur_riding(img):
    """Simulate motion blur from riding a bike."""
    size = random.choice([7, 9, 11, 15])
    # Horizontal motion blur (riding forward)
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


def diagonal_motion_blur(img):
    """Simulate turning/looking sideways while riding."""
    size = random.choice([7, 9, 11])
    kernel = np.zeros((size, size))
    # Diagonal blur
    for i in range(size):
        kernel[i, i] = 1
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)


def golden_hour(img):
    """Simulate sunset/golden hour warm tint."""
    warm = img.astype(np.float32)
    # Warm orange/golden tint
    warm[:, :, 2] = np.clip(warm[:, :, 2] * random.uniform(1.2, 1.5), 0, 255)  # Red
    warm[:, :, 1] = np.clip(warm[:, :, 1] * random.uniform(1.0, 1.2), 0, 255)  # Green
    warm[:, :, 0] = np.clip(warm[:, :, 0] * random.uniform(0.7, 0.9), 0, 255)  # Blue down
    # Slightly brighten
    warm *= random.uniform(1.0, 1.2)
    return np.clip(warm, 0, 255).astype(np.uint8)


def harsh_sunlight(img):
    """Simulate harsh midday sun with overexposure and shadows."""
    harsh = img.astype(np.float32)
    h, w = harsh.shape[:2]
    # Overexpose top portion (sun above)
    gradient = np.linspace(1.5, 0.8, h).reshape(h, 1, 1)
    harsh = harsh * gradient
    # Add bright spots
    for _ in range(random.randint(2, 5)):
        cx, cy = random.randint(0, w - 1), random.randint(0, h // 3)
        radius = random.randint(20, 50)
        mask_bright = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask_bright, (cx, cy), radius, 1.0, -1)
        mask_bright = cv2.GaussianBlur(mask_bright, (31, 31), 0)
        for c in range(3):
            harsh[:, :, c] += mask_bright * random.randint(50, 100)
    return np.clip(harsh, 0, 255).astype(np.uint8)


def heavy_shadow(img):
    """Simulate heavy shadows (tree shade, buildings)."""
    shadow = img.astype(np.float32)
    h, w = shadow.shape[:2]
    # Create random shadow patches
    mask = np.ones((h, w), dtype=np.float32)
    num_shadows = random.randint(2, 4)
    for _ in range(num_shadows):
        pts = []
        for _ in range(random.randint(3, 6)):
            pts.append([random.randint(0, w), random.randint(0, h)])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, random.uniform(0.3, 0.6))
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    for c in range(3):
        shadow[:, :, c] *= mask
    return np.clip(shadow, 0, 255).astype(np.uint8)


def simulate_mask(img):
    """Simulate face mask by adding a dark/colored band on lower face region."""
    masked = img.copy()
    h, w = masked.shape[:2]
    # Mask covers roughly lower 30-45% of face area (middle of image)
    y_start = int(h * random.uniform(0.50, 0.60))
    y_end = int(h * random.uniform(0.75, 0.85))
    x_start = int(w * random.uniform(0.15, 0.25))
    x_end = int(w * random.uniform(0.75, 0.85))
    # Random mask color (white, black, blue, green)
    colors = [
        (255, 255, 255),  # White surgical mask
        (40, 40, 40),     # Black mask
        (180, 130, 70),   # Blue surgical
        (80, 130, 80),    # Green surgical
        (200, 180, 160),  # Beige/skin tone
    ]
    color = random.choice(colors)
    # Create mask overlay with some transparency
    overlay = masked.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)
    alpha = random.uniform(0.5, 0.8)
    masked = cv2.addWeighted(overlay, alpha, masked, 1 - alpha, 0)
    return masked


def simulate_goggles(img):
    """Simulate sunglasses/goggles by adding dark band on eye region."""
    goggled = img.copy()
    h, w = goggled.shape[:2]
    # Goggles cover eye region (upper-middle of face)
    y_start = int(h * random.uniform(0.25, 0.35))
    y_end = int(h * random.uniform(0.40, 0.50))
    x_start = int(w * random.uniform(0.10, 0.20))
    x_end = int(w * random.uniform(0.80, 0.90))
    # Dark tinted goggles/sunglasses
    tints = [
        (20, 20, 20),     # Black
        (30, 30, 50),     # Dark blue tint
        (40, 25, 15),     # Brown tint
        (20, 40, 40),     # Dark green
        (50, 30, 60),     # Purple tint
    ]
    color = random.choice(tints)
    overlay = goggled.copy()
    # Rounded rectangle for goggles shape
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)
    # Add slight reflection shine
    shine_y = y_start + (y_end - y_start) // 4
    cv2.line(overlay, (x_start + 10, shine_y), (x_end - 10, shine_y),
             (180, 180, 200), 1)
    alpha = random.uniform(0.6, 0.85)
    goggled = cv2.addWeighted(overlay, alpha, goggled, 1 - alpha, 0)
    return goggled


def composite_blend(img1, img2):
    """Blend two images together to create a composite realistic scene."""
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    # Random blend ratio
    alpha = random.uniform(0.6, 0.85)
    blended = cv2.addWeighted(img1, alpha, img2_resized, 1 - alpha, 0)
    return blended


def random_occlusion(img):
    """Add random occlusion (hand, object blocking part of view)."""
    occluded = img.copy()
    h, w = occluded.shape[:2]
    # Random dark patch (simulating hand/object)
    num_patches = random.randint(1, 3)
    for _ in range(num_patches):
        p_w = random.randint(w // 8, w // 3)
        p_h = random.randint(h // 8, h // 3)
        x = random.randint(0, w - p_w)
        y = random.randint(0, h - p_h)
        # Skin-tone or dark object color
        colors = [(80, 60, 50), (120, 90, 70), (60, 50, 40), (30, 30, 30)]
        color = random.choice(colors)
        overlay = occluded.copy()
        cv2.rectangle(overlay, (x, y), (x + p_w, y + p_h), color, -1)
        alpha = random.uniform(0.3, 0.6)
        occluded = cv2.addWeighted(overlay, alpha, occluded, 1 - alpha, 0)
    return occluded


def dust_dirt(img):
    """Simulate dusty/dirty lens (Indian road conditions)."""
    dusty = img.copy()
    h, w = dusty.shape[:2]
    # Add brownish haze
    overlay = np.full_like(dusty, (130, 160, 180), dtype=np.uint8)  # Sandy color
    alpha = random.uniform(0.1, 0.25)
    dusty = cv2.addWeighted(overlay, alpha, dusty, 1 - alpha, 0)
    # Add random dust specks
    for _ in range(random.randint(30, 100)):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        r = random.randint(1, 3)
        cv2.circle(dusty, (x, y), r, (random.randint(140, 200),
                   random.randint(130, 180), random.randint(100, 160)), -1)
    return dusty


def streetlight_glow(img):
    """Simulate sodium streetlight glow (night riding in city)."""
    glow = img.astype(np.float32)
    # First darken (night)
    glow *= random.uniform(0.25, 0.45)
    # Add orange-yellow streetlight tint
    glow[:, :, 2] += random.randint(30, 60)   # Red
    glow[:, :, 1] += random.randint(20, 40)   # Green (for orange)
    glow[:, :, 0] += random.randint(0, 10)    # Minimal blue
    # Add a glow spot (light source)
    h, w = glow.shape[:2]
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(0, h // 3)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), random.randint(40, 80), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    for c in range(3):
        glow[:, :, c] += mask * random.randint(40, 80)
    return np.clip(glow, 0, 255).astype(np.uint8)


def headlight_glare(img):
    """Simulate oncoming headlight glare (night riding)."""
    glare = img.astype(np.float32)
    h, w = glare.shape[:2]
    # Darken first
    glare *= random.uniform(0.3, 0.5)
    # Add bright headlight spot
    cx = random.randint(w // 3, 2 * w // 3)
    cy = random.randint(h // 3, 2 * h // 3)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), random.randint(30, 60), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (71, 71), 0)
    glare += np.stack([mask, mask, mask], axis=-1) * random.randint(150, 250)
    return np.clip(glare, 0, 255).astype(np.uint8)


def color_cast(img):
    """Random color temperature shift (warm/cool)."""
    cast = img.astype(np.float32)
    # Random channel scaling
    for c in range(3):
        cast[:, :, c] *= random.uniform(0.7, 1.3)
    return np.clip(cast, 0, 255).astype(np.uint8)


def jpeg_artifact(img):
    """Simulate low-quality JPEG compression artifacts."""
    quality = random.randint(10, 30)
    _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def perspective_warp(img):
    """Random perspective distortion (different camera angles)."""
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.15)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [random.randint(0, margin), random.randint(0, margin)],
        [w - random.randint(0, margin), random.randint(0, margin)],
        [random.randint(0, margin), h - random.randint(0, margin)],
        [w - random.randint(0, margin), h - random.randint(0, margin)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ─── Core augmentation with geometric transforms ───
def base_geometric(img):
    """Apply random geometric transforms as base."""
    aug = img.copy()
    h, w = aug.shape[:2]

    # Random flip
    if random.random() > 0.5:
        aug = cv2.flip(aug, 1)

    # Random rotation
    angle = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random crop
    cf = random.uniform(0.7, 0.95)
    ch, cw = int(h * cf), int(w * cf)
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    aug = aug[y:y+ch, x:x+cw]
    aug = cv2.resize(aug, (w, h))

    return aug


# ═══════════════════════════════════════════════════
#  AUGMENTATION PIPELINE
# ═══════════════════════════════════════════════════

# All real-world augmentation functions grouped by category
AUGMENT_FUNCTIONS = {
    # Lighting conditions
    "night": night_mode,
    "dark_lowlight": dark_lowlight,
    "golden_hour": golden_hour,
    "harsh_sun": harsh_sunlight,
    "shadow": heavy_shadow,
    "streetlight": streetlight_glow,
    "headlight": headlight_glare,

    # Weather
    "rain": rain_effect,
    "fog": fog_haze,
    "dust": dust_dirt,

    # Motion
    "motion_blur": motion_blur_riding,
    "diagonal_blur": diagonal_motion_blur,

    # Accessories
    "mask": simulate_mask,
    "goggles": simulate_goggles,

    # Camera effects
    "occlusion": random_occlusion,
    "color_cast": color_cast,
    "jpeg_quality": jpeg_artifact,
    "perspective": perspective_warp,
}


def augment_image(img, all_images_in_class):
    """Apply 2-3 random augmentations to create a realistic variant."""
    aug = base_geometric(img)

    # Pick 2-3 random augmentation effects
    num_effects = random.randint(2, 3)
    chosen = random.sample(list(AUGMENT_FUNCTIONS.keys()), num_effects)

    for effect_name in chosen:
        func = AUGMENT_FUNCTIONS[effect_name]
        if effect_name == "composite" and all_images_in_class:
            other = random.choice(all_images_in_class)
            aug = composite_blend(aug, other)
        else:
            aug = func(aug)

    return aug, "+".join(chosen)


# ═══════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════

def load_originals(folder):
    """Load all original images from a folder."""
    images = []
    for f in sorted(os.listdir(folder)):
        if f.startswith('.') or f.startswith('rw_'):
            continue
        if not f.lower().endswith(VALID_EXT):
            continue
        if os.path.isdir(os.path.join(folder, f)):
            continue
        img = cv2.imread(os.path.join(folder, f))
        if img is not None:
            images.append((f, img))
    return images


def clean_previous_augments(folder):
    """Remove previously generated augments."""
    removed = 0
    for f in os.listdir(folder):
        if f.startswith('rw_') and f.lower().endswith(VALID_EXT):
            os.remove(os.path.join(folder, f))
            removed += 1
    return removed


def process_class(folder, class_name):
    """Generate augmented images for one class."""
    print(f"\n  📂 Processing: {class_name}/")

    # Clean old augments
    removed = clean_previous_augments(folder)
    if removed:
        print(f"     Cleaned {removed} old augments")

    # Load originals
    originals = load_originals(folder)
    print(f"     Original images: {len(originals)}")

    if len(originals) == 0:
        print("     ⚠️  No images found!")
        return 0

    # Calculate how many augments needed
    needed = TARGET_PER_CLASS - len(originals)
    if needed <= 0:
        print(f"     Already have {len(originals)} images, target is {TARGET_PER_CLASS}")
        return len(originals)

    print(f"     Need to generate: {needed} augmented images")
    print(f"     Augmentations per original: ~{needed // len(originals)}")

    # Load all images for compositing
    all_imgs = [img for _, img in originals]

    # Generate augments
    generated = 0
    effect_counts = {}

    while generated < needed:
        for orig_name, orig_img in originals:
            if generated >= needed:
                break

            aug_img, effects = augment_image(orig_img, all_imgs)
            out_name = f"rw_{generated:04d}_{effects.replace('+', '_')}.jpg"
            out_path = os.path.join(folder, out_name)

            cv2.imwrite(out_path, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            generated += 1

            # Track effects
            for e in effects.split("+"):
                effect_counts[e] = effect_counts.get(e, 0) + 1

            if generated % 50 == 0:
                print(f"     Generated: {generated}/{needed}")

    print(f"     ✅ Generated {generated} augmented images!")
    print(f"\n     Effect distribution:")
    for effect, count in sorted(effect_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 5)
        print(f"       {effect:<18} {count:>4} {bar}")

    return len(originals) + generated


# ─── Run ───
print("=" * 65)
print("  🏍️  HELIXO AI — Real-World Dataset Augmentation Engine")
print("=" * 65)
print(f"\n  Target: {TARGET_PER_CLASS} images per class ({TARGET_PER_CLASS * 2}+ total)")
print(f"  Augmentation types: {len(AUGMENT_FUNCTIONS)}")
print(f"  Effects: {', '.join(AUGMENT_FUNCTIONS.keys())}")

total_helmet = process_class(HELMET_DIR, "helmet")
total_no_helmet = process_class(NO_HELMET_DIR, "no_helmet")

print("\n" + "=" * 65)
print("  📊 FINAL DATASET SUMMARY")
print("=" * 65)
print(f"""
  Helmet images:      {total_helmet}
  No-helmet images:   {total_no_helmet}
  Total dataset:      {total_helmet + total_no_helmet}

  Augmentation effects applied:
    🌙 Night/Dark/Low-light (streetlights, headlights)
    🌧️  Rain, Fog, Dust (Indian road conditions)
    💨 Motion blur (riding simulation)
    🕶️  Goggles/Sunglasses overlay
    😷 Face mask overlay
    ☀️  Golden hour, Harsh sun, Heavy shadows
    📸 Camera effects (perspective, JPEG artifacts, occlusion)
    🎨 Color temperature shifts

  ✅ Dataset ready for training!
  ✅ All augments prefixed with 'rw_' for easy identification
""")
print("=" * 65)
