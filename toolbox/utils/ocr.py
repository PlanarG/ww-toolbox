from rapidocr import RapidOCR, EngineType
from PIL import Image, ImageFilter, ImageEnhance
from dataclasses import dataclass
from toolbox.utils.logger import logger

import math
import re
import numpy as np
import cv2
import random

engine = None

def setup_ocr():
    global engine
    engine = RapidOCR(
        params={
            "Det.engine_type": EngineType.OPENVINO,
            "Cls.engine_type": EngineType.OPENVINO,
            "Rec.engine_type": EngineType.OPENVINO,
        }
    )

@dataclass
class OCRResult:
    text: str
    box: tuple[int, int, int, int]
    confidence: float

def ocr(image: Image.Image, split: str = ' ') -> str:
    """
    Perform OCR on a PIL Image and return the detected text.
    Args:
        image (PIL.Image.Image): The image to process.
    Returns:
        str: The text detected in the image.
    """
    image = image.convert("L").convert("RGB").filter(ImageFilter.SHARPEN)
    result = engine(np.array(image))
    texts, boxes = result.txts, result.boxes

    predicted = ""
    last_right_bottom_y = 0

    try:
        for text, box in zip(texts, boxes):
            if box[0][1] >= last_right_bottom_y - 2:
                predicted += "\n"
            predicted += text + split
            last_right_bottom_y = box[2][1]
    except Exception:
        return ""

    return predicted.strip()

def ocr_pattern(image: Image.Image, pattern: str) -> list[OCRResult]:
    """
    Perform OCR on a PIL Image and return the detected texts with their boxes and confidence scores.
    Args:
        image (PIL.Image.Image): The image to process.
    Returns:
        list[OCRResult]: The detected texts with their boxes and confidence scores.
    """

    # ensure pattern is a legal regex pattern
    try:
        re.compile(pattern)
    except re.error:
        raise ValueError(f'Invalid pattern: {pattern}')

    image = image.convert("L").convert("RGB").filter(ImageFilter.SHARPEN)
    result = engine(np.array(image))
    texts, boxes, scores = result.txts, result.boxes, result.scores

    results = []
    try:
        for text, box, score in zip(texts, boxes, scores):
            if re.search(pattern, text):
                matched_text = re.search(pattern, text).group(0)
                results.append(OCRResult(matched_text, (box[0][0], box[0][1], box[2][0], box[2][1]), score))
    except Exception:
        return []

    return results

def match_single_object_template(
    query_img: Image.Image, 
    target_img: Image.Image, 
    match_threshold=0.3, 
    scale_range=(0.5, 2.0), 
    scale_steps=30, 
    debug=False
) -> tuple[int, int] | None:
    """
    Finds an object in a target image using multi-scale template matching on their Canny edge maps.
    This method is robust for objects with consistent shape and rotation, and can handle variations in scale.

    Args:
        query_img (PIL.Image.Image): The template image to find.
        target_img (PIL.Image.Image): The image to search within.
        match_threshold (float): The minimum correlation score to be considered a good match. Higher is better.
        scale_range (tuple): The range of scales (min_scale, max_scale) to search.
        scale_steps (int): The number of steps to iterate through within the scale range.
        debug (bool): Whether to display matching visualization.

    Returns:
        tuple[int, int] | None: Center coordinates (x, y) of the matched object or None if not found.
    """
    def preprocess_edges(gray: np.ndarray) -> np.ndarray:
        # Add a median blur to combat salt-and-pepper noise. A 5x5 kernel is used for robustness.
        blurred = cv2.medianBlur(gray, 5)
        equalized = cv2.equalizeHist(blurred)
        edges = cv2.Canny(equalized, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges

    query_gray = np.array(query_img.convert('L'))
    target_gray = np.array(target_img.convert('L'))

    query_edges = preprocess_edges(query_gray)
    if np.sum(query_edges) == 0:
        return None

    tH, tW = query_edges.shape
    best_match = None

    for scale in np.linspace(scale_range[0], scale_range[1], scale_steps)[::-1]:
        resized_target = cv2.resize(
            target_gray,
            (int(target_gray.shape[1] * scale), int(target_gray.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR
        )

        if resized_target.shape[0] < tH or resized_target.shape[1] < tW:
            break

        target_edges = preprocess_edges(resized_target)
        result = cv2.matchTemplate(target_edges, query_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if best_match is None or max_val > best_match[0]:
            best_match = (max_val, max_loc, scale)

    if best_match and best_match[0] >= match_threshold:
        max_val, max_loc, scale = best_match
        top_left_unscaled = (int(max_loc[0] / scale), int(max_loc[1] / scale))
        w_unscaled = int(tW / scale)
        h_unscaled = int(tH / scale)
        center_x = top_left_unscaled[0] + w_unscaled // 2
        center_y = top_left_unscaled[1] + h_unscaled // 2

        if debug:
            target_vis = np.array(target_img.convert('RGB'))
            target_vis = cv2.cvtColor(target_vis, cv2.COLOR_RGB2BGR)
            bottom_right = (top_left_unscaled[0] + w_unscaled, top_left_unscaled[1] + h_unscaled)
            cv2.rectangle(target_vis, top_left_unscaled, bottom_right, (0, 255, 0), 2)
            cv2.circle(target_vis, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.imshow(f"Template Match (Score: {max_val:.2f}, Scale: {scale:.2f})", target_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return (center_x, center_y)

    return None

def _detect_rectangles_raw(
    pil_image: Image.Image,
    aspect_ratio_range=(0.8, 1.2),
    area_range=(200, 1e3),
    canny_thresh_low=50,
    canny_thresh_high=100
) -> list[tuple[int, int, int, int]]:
    """Detects raw rectangular regions from a PIL image without merging."""
    image = np.array(pil_image)
    if image.ndim == 2:
        gray = image.copy()
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_thresh_low, canny_thresh_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (area_range[0] < area < area_range[1]):
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            def angle(pt1, pt2, pt0):
                dx1 = pt1[0] - pt0[0]
                dy1 = pt1[1] - pt0[1]
                dx2 = pt2[0] - pt0[0]
                dy2 = pt2[1] - pt0[1]
                return (dx1 * dx2 + dy1 * dy2) / (math.hypot(dx1, dy1) * math.hypot(dx2, dy2) + 1e-8)

            pts = approx.reshape(-1, 2)
            max_cos = max(abs(angle(pts[i], pts[(i + 2) % 4], pts[(i + 1) % 4])) for i in range(4))
            if max_cos > 0.3:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h if h > 0 else 0
            if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                continue
            
            rects.append((x, y, w, h))
            
    return rects

def detect_and_merge_rectangles_pil(
    pil_image: Image.Image,
    aspect_ratio_range=(0.8, 1.2),
    area_range=(300, 1e3),
    iou_threshold=0.25,
    num_perturbations=50,
    brightness_threshold=100,
    bright_area_ratio_threshold=0.35,
    debug=False
) -> list[tuple[int, int, int, int]]:
    """
    Detects and merges all rectangular (or rounded rectangular) regions from a PIL image.
    This version is more robust by applying random perturbations to the image.

    Args:
        pil_image (PIL.Image): The input image.
        aspect_ratio_range (tuple): The aspect ratio range (min_ratio, max_ratio).
        area_range (tuple): The area range (min_area, max_area).
        iou_threshold (float): The IoU threshold for merging rectangles.
        num_perturbations (int): The number of random perturbations to apply.
        brightness_threshold (int): The threshold for bright pixels.
        bright_area_ratio_threshold (float): The threshold for the ratio of bright pixels.
        debug (bool): Whether to visualize the results.

    Returns:
        list[tuple[int, int, int, int]]: A list of merged rectangles as (x, y, w, h).
    """
    # Scale image to target width 512 while maintaining aspect ratio
    orig_width, orig_height = pil_image.size
    scale = 512 / orig_width
    target_height = int(orig_height * scale)
    scaled_image = pil_image.resize((512, target_height), Image.Resampling.BICUBIC)

    all_rects = []

    # Original image with default canny thresholds
    all_rects.extend(_detect_rectangles_raw(scaled_image, aspect_ratio_range, area_range))

    # Perturbed images
    for _ in range(num_perturbations):
        current_image = scaled_image.copy()

        # Apply a chain of random enhancements
        enhancers = [
            (ImageEnhance.Brightness, random.uniform(0.8, 1.2)),
            (ImageEnhance.Contrast, random.uniform(0.8, 1.5)),
            (ImageEnhance.Sharpness, random.uniform(0.7, 1.3)),
        ]
        random.shuffle(enhancers)
        
        for enhancer, factor in enhancers:
            current_image = enhancer(current_image).enhance(factor)
        
        # Convert to numpy for cv2-based preprocessing
        np_image = np.array(current_image.convert('L'))

        # Apply a random preprocessing strategy before Canny detection
        strategy = random.choice(['blur', 'morph', 'equalize', 'none'])

        if strategy == 'blur':
            blur_type = random.choice(['gaussian', 'median', 'bilateral'])
            if blur_type == 'gaussian':
                kernel_size = random.choice([3, 5, 7])
                processed_image = cv2.GaussianBlur(np_image, (kernel_size, kernel_size), 0)
            elif blur_type == 'median':
                kernel_size = random.choice([3, 5, 7])
                processed_image = cv2.medianBlur(np_image, kernel_size)
            else:  # bilateral
                processed_image = cv2.bilateralFilter(np_image, d=9, sigmaColor=75, sigmaSpace=75)
        elif strategy == 'morph':
            op = random.choice([cv2.MORPH_OPEN, cv2.MORPH_CLOSE])
            kernel = np.ones((3, 3), np.uint8)
            processed_image = cv2.morphologyEx(np_image, op, kernel)
        elif strategy == 'equalize':
            processed_image = cv2.equalizeHist(np_image)
        else:  # 'none'
            processed_image = np_image
        
        pil_processed = Image.fromarray(processed_image)

        # Use randomized Canny thresholds for more robustness
        canny_low = random.randint(30, 70)
        canny_high = random.randint(80, 200)

        all_rects.extend(_detect_rectangles_raw(
            pil_processed, aspect_ratio_range, area_range, canny_low, canny_high
        ))

    # Merge all collected rectangles
    merged = merge_rectangles(all_rects, iou_threshold)

    merged.sort(key=lambda x: (x[1], x[0]))
    i = 0
    while i < len(merged) - 1:
        if abs(merged[i][1] - merged[i+1][1]) < 5 and merged[i][0] > merged[i+1][0]:
            merged[i], merged[i+1] = merged[i+1], merged[i]
            if i > 0:
                i -= 1
        else:
            i += 1

    # Scale coordinates back to original image size
    scale_back = orig_width / 512
    merged = [(round(x * scale_back), round(y * scale_back), 
              round(w * scale_back), round(h * scale_back)) for x, y, w, h in merged]

    # Filter rectangles based on internal brightness
    final_rects = []
    for x, y, w, h in merged:
        if w <= 0 or h <= 0:
            continue
        
        rect_img = pil_image.crop((x, y, x + w, y + h)).convert('L')
        rect_data = np.array(rect_img)
        
        bright_pixels = np.sum(rect_data > brightness_threshold)
        total_pixels = w * h
        
        bright_ratio = bright_pixels / total_pixels
        
        if bright_ratio <= bright_area_ratio_threshold:
            final_rects.append((x, y, w, h))

    if debug:
        vis_image = np.array(pil_image.convert('RGB'))
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        # Draw all merged rectangles in red
        for x, y, w, h in merged:
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # Draw final, filtered rectangles in green
        for x, y, w, h in final_rects:
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Merged & Filtered Rectangles", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_rects

def merge_rectangles(rects: list[tuple[int, int, int, int]], iou_thresh=0.2) -> list[tuple[int, int, int, int]]:
    """
    Merges a group of overlapping or adjacent rectangles based on IoU.

    Args:
        rects (list[tuple[int, int, int, int]]): The input rectangles (x, y, w, h).
        iou_thresh (float): The IoU threshold for merging.

    Returns:
        list[tuple[int, int, int, int]]: A list of merged rectangles.
    """
    if not rects:
        return []

    rects = np.array(rects).astype(float).tolist()
    merged = []

    while rects:
        base = rects.pop(0)
        i = 0
        while i < len(rects):
            if compute_iou(base, rects[i]) > iou_thresh:
                x1 = min(base[0], rects[i][0])
                y1 = min(base[1], rects[i][1])
                x2 = max(base[0] + base[2], rects[i][0] + rects[i][2])
                y2 = max(base[1] + base[3], rects[i][1] + rects[i][3])
                base = [x1, y1, x2 - x1, y2 - y1]
                rects.pop(i)
                i = 0  # Restart scan since base has changed
            else:
                i += 1
        merged.append(tuple(map(int, base)))

    return merged

def compute_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """
    Computes the Intersection over Union (IoU) between two rectangles.
    
    Args:
        a (tuple[int, int, int, int]): The first rectangle (x, y, w, h).
        b (tuple[int, int, int, int]): The second rectangle (x, y, w, h).
        
    Returns:
        float: The IoU score, a value between 0 and 1.
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0

def _change_hue_cv2(image: Image.Image, hue_shift: int) -> Image.Image:
    """
    Shifts the hue of an image using OpenCV. For testing purposes.
    hue_shift: angle in degrees to shift hue (0-180 for OpenCV).
    """
    rgb_img = np.array(image.convert('RGB'))
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    # The H channel in OpenCV is in the range [0, 179]
    hsv_img[..., 0] = (hsv_img[..., 0].astype(int) + hue_shift) % 180
    new_rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return Image.fromarray(new_rgb_img)

def _add_salt_and_pepper_noise(image: Image.Image, amount=0.05) -> Image.Image:
    """
    Adds salt and pepper noise to a PIL image. For testing purposes.
    """
    img_np = np.array(image.convert('RGB'))
    h, w, _ = img_np.shape
    
    # Salt noise (white pixels)
    num_salt = int(amount * h * w * 0.5)
    ys_salt = np.random.randint(0, h, num_salt)
    xs_salt = np.random.randint(0, w, num_salt)
    img_np[ys_salt, xs_salt, :] = 255

    # Pepper noise (black pixels)
    num_pepper = int(amount * h * w * 0.5)
    ys_pepper = np.random.randint(0, h, num_pepper)
    xs_pepper = np.random.randint(0, w, num_pepper)
    img_np[ys_pepper, xs_pepper, :] = 0
    
    return Image.fromarray(img_np)

if __name__ == "__main__":
    setup_ocr()
    img = Image.open("tests/test-level.png")
    print(ocr(img, split=""))
    print(ocr_pattern(img, "\d+"))
    img = Image.open("tests/test-rare.png")
    print(ocr(img))
    # --- Rectangle detection tests ---
    print("\n--- Testing detect_and_merge_rectangles_pil ---")
    try:
        rect_img = Image.open("tests/test-rect2.png")
        print("Running rectangle detection test...")
        rects = detect_and_merge_rectangles_pil(rect_img, debug=True)
        print(f"  Found {len(rects)} rectangles.")
        for i, r in enumerate(rects):
            print(f"    Rect {i+1}: {r}")
            
    except FileNotFoundError:
        print("\nSkipping rectangle detection tests: 'tests/test-rect2.png' not found.")
        
    # --- Template matching tests ---
    print("\n--- Testing match_single_object_template ---")
    try:
        template_img_orig = Image.open("tests/test-template.png")
        match_img_orig = Image.open("tests/test-match.png")

        print("Running template matching test...")
        coords = match_single_object_template(template_img_orig, match_img_orig, debug=True)
        print(f"  Result: {coords}")

        # --- Hue shift test ---
        print("\nRunning template matching test with hue shift (should still work)...")
        hue_shifted_img = _change_hue_cv2(match_img_orig, 90) # 90 degree shift
        coords_hue = match_single_object_template(template_img_orig, hue_shifted_img, debug=True)
        print(f"  Result with hue shift: {coords_hue}")

        # --- Noise test ---
        print("\nRunning template matching test with noise...")
        noisy_img = _add_salt_and_pepper_noise(match_img_orig, amount=0.02)
        coords_noise = match_single_object_template(template_img_orig, noisy_img, debug=True)
        print(f"  Result with noise: {coords_noise}")
            
    except FileNotFoundError:
        print("\nSkipping template matching tests: 'tests/test-template.png' or 'tests/test-match.png' not found.")
