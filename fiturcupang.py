#!/usr/bin/env python3
"""
Dependencies:
    pip install opencv-python-headless scikit-image pandas numpy tqdm

Usage:
    python extract__features.py /path/to/root_dir output_features.csv
Where root_dir contains subfolders per class:
    root_dir/crowntail/*.jpg
    root_dir/doubletail/*.png
    ...
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage import color, filters, feature, measure, morphology, exposure
from skimage.util import img_as_ubyte
from skimage.measure import regionprops, label
from tqdm import tqdm

# -----------------------------
# Helper feature functions
# -----------------------------
def read_image(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    # If PNG with alpha, composite on white background
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
        img_rgb = (img[:, :, :3].astype(float) * alpha[..., None] + bg.astype(float) * (1-alpha[..., None])).astype(np.uint8)
        img = img_rgb
    # if grayscale -> convert to BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # ensure BGR uint8
    if img.dtype != np.uint8:
        img = img_as_ubyte(img)
    return img

def simple_segment_fish(bgr):
    """Segmentasi sederhana: grayscale -> gaussian -> otsu -> morph -> pick largest region"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # equalize for robustness
    gray = exposure.equalize_adapthist(gray)  # float in 0..1
    gray_u = img_as_ubyte(gray)
    blur = cv2.GaussianBlur(gray_u, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert if background dark/bright heuristics
    # ensure object is white on black for morphology
    white_ratio = (th==255).mean()
    if white_ratio > 0.9:  # probably background is white -> invert
        th = cv2.bitwise_not(th)
    # morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    # label and choose largest region
    lbl = label(th > 0)
    if lbl.max() == 0:
        return None  # segmentation failed
    props = regionprops(lbl)
    # pick region with largest area
    largest = max(props, key=lambda p: p.area)
    mask = (lbl == largest.label).astype(np.uint8) * 255
    # fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # final small cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # sanity check - if mask too small, consider failure
    if mask.sum() < 50:
        return None
    return mask

def contour_shape_features(mask):
    """Compute contour-based shape features from binary mask (uint8 0/255)"""
    res = {}
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # zeros
        keys = ["area","perimeter","aspect_ratio","extent","solidity","equivalent_diameter","eccentricity","bbox_area"]
        for k in keys:
            res[k] = 0.0
        return res
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x,y,w,h = cv2.boundingRect(c)
    bbox_area = w*h if (w*h)>0 else 1
    aspect_ratio = float(w)/float(h) if h>0 else 0.0
    rect_area = bbox_area
    extent = float(area)/rect_area if rect_area>0 else 0.0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area>0 else 0.0
    equi_diam = np.sqrt(4*area/np.pi) if area>0 else 0.0
    # eccentricity via regionprops:
    lbl = label(mask>0)
    props = regionprops(lbl)
    ecc = 0.0
    if props:
        ecc = props[0].eccentricity
    res.update({
        "area": float(area),
        "perimeter": float(perimeter),
        "aspect_ratio": float(aspect_ratio),
        "extent": float(extent),
        "solidity": float(solidity),
        "equivalent_diameter": float(equi_diam),
        "eccentricity": float(ecc),
        "bbox_area": float(bbox_area)
    })
    return res

def hu_moments(mask_or_gray):
    """Return 7 Hu moments (log-transformed absolute values)"""
    # require single-channel image (mask or gray)
    if mask_or_gray.ndim == 3:
        im = cv2.cvtColor(mask_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        im = mask_or_gray
    moments = cv2.moments(im.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    # log transform to reduce dynamic range, keep sign via -sign*log10(abs)
    hu_log = []
    for h in hu:
        if h == 0:
            hu_log.append(0.0)
        else:
            hu_log.append(-np.sign(h) * np.log10(abs(h)))
    return {"hu1":hu_log[0],"hu2":hu_log[1],"hu3":hu_log[2],"hu4":hu_log[3],
            "hu5":hu_log[4],"hu6":hu_log[5],"hu7":hu_log[6]}

def color_features(bgr, mask=None, nbins=8):
    """Compute mean/std per channel (BGR & HSV) and normalized histograms (per channel).
       Returns flattened dict."""
    res = {}
    # masked mean/std
    if mask is None:
        roi = bgr
        mask_bool = None
    else:
        mask_bool = mask.astype(bool)
        roi = bgr.copy()
    # RGB (OpenCV uses BGR)
    chans = cv2.split(bgr)
    for i, ch_name in enumerate(["B","G","R"]):
        ch = chans[i]
        if mask is None:
            mmean = float(ch.mean())
            mstd = float(ch.std())
        else:
            vals = ch[mask_bool]
            if vals.size==0:
                mmean = 0.0; mstd = 0.0
            else:
                mmean = float(vals.mean()); mstd = float(vals.std())
        res[f"{ch_name}_mean"] = mmean
        res[f"{ch_name}_std"] = mstd
        # histogram
        hist = cv2.calcHist([ch],[0], None if mask is None else mask, [nbins], [0,256])
        hist = hist.flatten()
        if hist.sum()>0:
            hist = hist / hist.sum()
        for j, v in enumerate(hist):
            res[f"{ch_name}_hist_{j}"] = float(v)
    # HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    for arr, name in zip([h,s,v], ["H","S","V"]):
        if mask is None:
            mmean = float(arr.mean()); mstd = float(arr.std())
        else:
            vals = arr[mask_bool]
            if vals.size==0:
                mmean = 0.0; mstd = 0.0
            else:
                mmean = float(vals.mean()); mstd = float(vals.std())
        res[f"{name}_mean"] = mmean
        res[f"{name}_std"] = mstd
        # histogram
        hist = cv2.calcHist([arr],[0], None if mask is None else mask, [nbins], [0,256])
        hist = hist.flatten()
        if hist.sum()>0:
            hist = hist / hist.sum()
        for j,v in enumerate(hist):
            res[f"{name}_hist_{j}"] = float(v)
    return res

def glcm_texture_features(gray, mask=None, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64):
    """Compute GLCM props: contrast, dissimilarity, homogeneity, energy, correlation, ASM.
       We'll quantize to `levels` to speed up."""
    if gray.dtype != np.uint8:
        gray_u = img_as_ubyte(gray)
    else:
        gray_u = gray
    # apply mask by cropping the region of interest to reduce background influence
    if mask is not None:
        ys, xs = np.where(mask>0)
        if ys.size==0:
            roi = gray_u
        else:
            minr, maxr = ys.min(), ys.max()
            minc, maxc = xs.min(), xs.max()
            roi = gray_u[minr:maxr+1, minc:maxc+1]
            if roi.size==0:
                roi = gray_u
    else:
        roi = gray_u
    # quantize
    if levels < 256:
        roi_q = np.floor((roi.astype(np.float32)/256.0) * levels).astype(np.uint8)
    else:
        roi_q = roi
    # if ROI too small, return zeros
    if roi_q.size < 16:
        return {f"glcm_{p}":0.0 for p in ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]}
    # compute GLCM using skimage
    from skimage.feature import graycomatrix, graycoprops
    try:
        glcm = graycomatrix(roi_q, distances=distances, angles=angles, levels=max(levels,256), symmetric=True, normed=True)
    except Exception:
        # fallback: use levels=levels
        glcm = graycomatrix(roi_q, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    props = {}
    for prop in ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]:
        vals = graycoprops(glcm, prop)
        # average across distances & angles
        props[f"glcm_{prop}"] = float(np.nanmean(vals))
    return props

def lbp_features(gray, mask=None, P=8, R=1, n_bins=10):
    """Local Binary Pattern histogram (normalized)."""
    if gray.dtype != np.uint8:
        gray_u = img_as_ubyte(gray)
    else:
        gray_u = gray
    lbp = feature.local_binary_pattern(gray_u, P, R, method='uniform')
    if mask is not None:
        data = lbp[mask>0]
    else:
        data = lbp.ravel()
    if data.size == 0:
        hist = np.zeros(n_bins)
    else:
        # number of unique patterns for 'uniform' is P+2 usually; we'll compute histogram over range
        max_val = int(lbp.max()) + 1
        hist, _ = np.histogram(data, bins=n_bins, range=(0, max(10, max_val)))
        if hist.sum()>0:
            hist = hist / hist.sum()
    res = {}
    for i,v in enumerate(hist):
        res[f"lbp_hist_{i}"] = float(v)
    return res

# -----------------------------
# Main extraction for one image
# -----------------------------
def extract_features_for_image(path):
    img = read_image(path)  # BGR uint8
    h0, w0 = img.shape[:2]
    # try segmentation
    mask = simple_segment_fish(img)
    # if segmentation failed, use central crop heuristic
    if mask is None:
        # fallback: whole image as mask
        mask = np.ones((h0,w0), dtype=np.uint8)*255
    # compute features
    feat = {}
    feat["filename"] = str(path)
    feat["label"] = Path(path).parent.name
    # color features
    cf = color_features(img, mask=mask, nbins=8)
    feat.update(cf)
    # shape features
    sf = contour_shape_features(mask)
    feat.update(sf)
    # hu moments
    hm = hu_moments(mask)
    feat.update(hm)
    # texture: GLCM
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = glcm_texture_features(gray, mask=mask, distances=[1,2], angles=[0, np.pi/4, np.pi/2], levels=64)
    feat.update(glcm)
    # LBP
    lbp = lbp_features(gray, mask=mask, P=8, R=1, n_bins=10)
    feat.update(lbp)
    # image size info
    feat["img_width"] = int(w0)
    feat["img_height"] = int(h0)
    feat["mask_pixel_count"] = int((mask>0).sum())
    return feat

# -----------------------------
# Walk dataset and save CSV
# -----------------------------
# -----------------------------
# Walk dataset and save CSV
# -----------------------------
def process_dataset(root_dir, out_csv, exts=(".jpg",".jpeg",".png",".tif",".bmp", ".jfif")):
    root = Path(root_dir)
    rows = []
    image_paths = []
    # collect image files under immediate subfolders (class folders)
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                image_paths.append(p)
    print(f"Found {len(image_paths)} images across {len(list(root.iterdir()))} folders.")
    for p in tqdm(image_paths):
        try:
            feat = extract_features_for_image(p)
            rows.append(feat)
        except Exception as e:
            print(f"Error processing {p}: {e}")
    if not rows:
        raise RuntimeError("No features extracted.")
    df = pd.DataFrame(rows)

    # --- FIX: tempatkan label di kolom paling kanan ---
    all_cols = list(df.columns)
    all_cols.remove("filename")
    all_cols.remove("label")
    cols = ["filename"] + all_cols + ["label"]   # label now last
    df = df[cols]
    # ---------------------------------------------------

    df.to_csv(out_csv, index=False)
    print(f"Saved features to {out_csv}")
    return df


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_betta_features.py /path/to/root_dir output.csv")
        sys.exit(1)
    root_dir = sys.argv[1]
    out_csv = sys.argv[2]
    df = process_dataset(root_dir, out_csv)
    print(df.head())

