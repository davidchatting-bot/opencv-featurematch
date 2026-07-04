const identityMatrix = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
];

const imageTransforms = [];
let mediaBoundingBox = null;

// Once a candidate match clears this many RANSAC inliers, stop searching further
// (temporally-nearer) candidates — an early exit to avoid an O(n^2) alignment
// search across the whole sequence when the nearest neighbour is already a
// confident match, which is true for the large majority of burst-sequence pairs.
const EARLY_EXIT_INLIER_THRESHOLD = 50;

let maskSegmentation = null;

// Real-time playback of the capture sequence, driven by each image's EXIF timestamp.
const PLAYBACK_END_PAUSE_MS = 3000;
const PLAYBACK_SPEED = 0.5; // 1 = real-time (matches original capture pace), 0.5 = half speed
let playbackSchedule = [];
let playbackStartMillis = 0;

// Used only for images with no recoverable EXIF timestamp, to keep the
// playback schedule well-formed (monotonically increasing) when falling back
// to alphabetical-filename ordering.
const FALLBACK_FRAME_SPACING_MS = 500;

// When the timeline hits a given image's own scheduled moment, it shows at
// full opacity for this long (real wall-clock ms, NOT scaled by
// PLAYBACK_SPEED — the "moment" is the crisp instant of the shot itself),
// centred on its scheduled time. Every image still shows faintly at all
// times via the constant low alpha in draw().
const MOMENT_MS = 200;

// 3D camera fly-through: one keyframe per aligned image, framing that image
// alone, timed to the exact same clock as playbackSchedule above — the camera
// is exactly centred on an image at the moment it becomes the current frame,
// then travels to be exactly framed on the next image by its due time.
let cameraKeyframes = [];

/**
 * Creates the foreground segmenter and waits until it's ready.
 * Returns a Promise that resolves when the segmenter is ready.
 */
async function createForegroundSegmenter() {
  maskSegmentation = new SelfieSegmentation({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@latest/${file}`;
    }
  });
  var options = {
    selfieMode: true,
    modelSelection: 0, // general
    effect: 'mask',
  };
  maskSegmentation.setOptions(options);

  // Wait for the WASM to load (simulate "blocking" until ready)
  await maskSegmentation.initialize();
}

// In setup(), use async/await to block until ready
async function setup() {
  // Use WEBGL so texture()/vertex(u,v) in drawImageWithHomography works
  canvas = createCanvas(2000, 800, WEBGL);
  frameRate(10);
  canvas.drop(onFileDropped);

  // Block until segmenter is ready
  await createForegroundSegmenter();

  // ensure texture UVs use normalized coordinates
  textureMode(NORMAL);

  processAnyAttachedMedia();
}

function onFileDropped(file) {
  const id = file.name;
  console.log("Dropped file: " + file.name);
  const div = upsertMedia(id);

  const originalImg = createImg(file.data, '', () => {
    originalImg.parent(div);
    originalImg.addClass('original');
    setImageTransform(originalImg.elt, identityMatrix);

    processImage(originalImg.elt, div);
  });
}

function generateLowResImage(imgElement, onloaded = () => {}) {
  let lowresImg = null;

  const lowresMaxPixels = 1024 * 768;
  if (imgElement.width * imgElement.height > lowresMaxPixels) { 
    const s = Math.sqrt(lowresMaxPixels / (imgElement.width * imgElement.height));

    const targetW = Math.round(imgElement.width * s);
    const targetH = Math.round(imgElement.height * s);
    
    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(imgElement, 0, 0, targetW, targetH);

    const dataUrl = canvas.toDataURL("image/jpeg", 1.0);
    canvas.width = 0;
    canvas.height = 0;

    lowresImg = createImg(dataUrl, '');
    lowresImg.elt.onload = onloaded;

    // Attach a 4x4 scaling transform (row-major)
    const invS = 1 / s;
    const scaleTransform = [
      invS, 0, 0, 0,
      0, invS, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ];
    setImageTransform(lowresImg.elt, scaleTransform);
  } else {
    lowresImg = new p5.Element(imgElement);
    setTimeout(onloaded, 0);

    // Attach identity transform (no scaling)
    setImageTransform(lowresImg.elt, [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ]);
  }

  return lowresImg;
}

function generateMask(imgElement, onloaded = () => {}) {
  let maskImg = createImg('', '');

  maskSegmentation.onResults(async (results) => {
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = results.segmentationMask.width;
      maskCanvas.height = results.segmentationMask.height;
      const ctx = maskCanvas.getContext('2d');
      
      // flip horizontally
      ctx.translate(maskCanvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(results.segmentationMask, 0, 0);

      // convert red-channel mask to greyscale (copy R to G and B)
      const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];     // red channel holds the mask value
        data[i]     = r;       // R (keep)
        data[i + 1] = r;       // G (copy from R)
        data[i + 2] = r;       // B (copy from R)
      }
      ctx.putImageData(imageData, 0, 0);
      setImageTransform(maskImg.elt, getImageTransformFromElement(imgElement));

      maskImg.elt.onload = onloaded;
      maskImg.elt.src = maskCanvas.toDataURL();
    });
    maskSegmentation.send({ image: imgElement });

    return(maskImg);
}

function processHomography(id) {
  const selector = '.background';
  const mediaCollection = select('#media')?.elt.querySelectorAll(selector);
  if (!mediaCollection || mediaCollection.length === 0) return;

  const n = mediaCollection.length;

  if (n === 1) {
    setImageTransform(mediaCollection[0].parentElement, identityMatrix);
    return;
  }

  // The newest image is the last one
  const image_b = mediaCollection[n - 1];

  // Skip if already aligned
  if (getImageTransformFromElement(image_b.parentElement)) return;

  // Try all previously aligned images and pick the best match by inlier count
  let bestInliers = 0;
  let bestT0B = null;
  let bestMatchId = null;

  for (let i = n - 2; i >= 0; i--) {
    const image_a = mediaCollection[i];
    const t0A = getImageTransformFromElement(image_a.parentElement);

    // Skip images that haven't been aligned yet
    if (!t0A) continue;

    console.log('*** trying align ', image_a.parentElement.id, image_b.parentElement.id);
    Align_img(image_a, image_b);

    const inlierCount = (good_inlier_matches && good_inlier_matches.size) ? good_inlier_matches.size() : 0;

    if (h && !h.empty() && h.data64F) {
      const check = isReasonableHomography(Array.from(h.data64F));
      console.log('Homography check:', check, 'inliers:', inlierCount);

      if (check.valid && inlierCount > bestInliers) {
        const tab = [
          h.data64F[0], h.data64F[1], 0, h.data64F[2],
          h.data64F[3], h.data64F[4], 0, h.data64F[5],
          0, 0, 1, 0,
          h.data64F[6], h.data64F[7], 0, h.data64F[8]
        ];

        const tAa = getImageTransformFromElement(image_a);
        const tBb = getImageTransformFromElement(image_b);
        const tBb_i = invertMatrix4x4(tBb);
        const tAB = multiplyMatrix4x4(multiplyMatrix4x4(tAa, tab), tBb_i);

        bestT0B = multiplyMatrix4x4(t0A, tAB);
        bestInliers = inlierCount;
        bestMatchId = image_a.parentElement.id;

        // Candidates are tried nearest-in-time first (i counts down from n-2),
        // so a confident match here is very likely the best one available —
        // stop searching rather than aligning against every earlier frame too.
        if (bestInliers >= EARLY_EXIT_INLIER_THRESHOLD) {
          console.log('Early exit: accepting', bestMatchId, 'with', bestInliers, 'inliers (>=', EARLY_EXIT_INLIER_THRESHOLD, ')');
          break;
        }
      } else if (!check.valid) {
        console.warn('Rejecting homography with', image_a.parentElement.id, ':', check.reason);
      }
    }
  }

  if (bestT0B) {
    console.log('Best match for', image_b.parentElement.id, ':', bestMatchId, 'with', bestInliers, 'inliers');
    setImageTransform(image_b.parentElement, bestT0B);
  } else {
    console.warn('No valid homography found for', image_b.parentElement.id);
  }
}

// cache for converted images (HTMLImageElement -> p5.Graphics)
const textureCache = new WeakMap();

function getTextureFromElement(el) {
  if (!el) return null;

  // Use natural pixel dimensions, not the CSS-rendered box size — el.width/height
  // reflect layout (and go wrong if #media's display/sizing CSS changes), while
  // naturalWidth/naturalHeight are the actual decoded image dimensions.
  const w = el.naturalWidth || el.width;
  const h = el.naturalHeight || el.height;

  // check cache first
  if (textureCache.has(el)) {
    const cached = textureCache.get(el);
    // check if image size changed (unlikely but safe)
    if (cached.width === w && cached.height === h) {
      return cached;
    }
    // size changed, remove old and recreate
    cached.remove();
  }

  // convert HTMLImageElement to p5.Graphics
  const g = createGraphics(w, h);
  g.drawingContext.drawImage(el, 0, 0);
  textureCache.set(el, g);
  return g;
}

// draw a textured quad: srcImg projected by homography Hproj into target image space (targetIndex)
function drawProjectedImage(srcImg, x, y, Hproj, zDepth = 0) {
  if (!srcImg || !Hproj) return;
  
  const img = getTextureFromElement(srcImg);
  if (!img) return;
  
  const w = img.width, h = img.height;
  const corners = [0,0,w,0,w,h,0,h];
  // project corners into target image pixel coords (corners is a flat array [x0,y0,...])
  const dst = [];
  for (let i = 0; i < corners.length; i += 2) {
    const p = applyTransform4x4(corners[i], corners[i + 1], Hproj) || [0, 0];
    dst.push(p[0]+x, p[1]+y);
  }
  // draw textured polygon in WEBGL using normalized texture coords (0..1)
  push();
    noStroke();
    texture(img);
    beginShape();
      // vertex(x, y, z, u, v)
      vertex(dst[0], dst[1], zDepth, 0, 0);
      vertex(dst[2], dst[3], zDepth, 1, 0);
      vertex(dst[4], dst[5], zDepth, 1, 1);
      vertex(dst[6], dst[7], zDepth, 0, 1);
    endShape(CLOSE);
  pop();
}

function upsertMedia(id) {
  if (!id) return null;

  let container = select('#media');
  if (!container) return null;

  const found = container.elt.querySelector('#' + id);
  if (found) return select('#' + id); // use p5.select to return a p5.Element

  const d = createDiv('');
  d.id(id);
  d.parent(container);
  return d;
}

/**
 * Returns the bounding box (in screen coordinates) that contains all media elements,
 * with their transforms applied (using imageTransforms).
 * @returns {{left: number, top: number, right: number, bottom: number}|null}
 */
function getBoundingBox(selector, indices) {
  const mediaElement = select('#media')?.elt;
  if (!mediaElement) return null;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  for (const i of indices) {
    const transform = getImageTransformFromElement(mediaElement.children[i], true);
    if (!transform) continue;

    const image = mediaElement.children[i].querySelector(selector);
    if (!image) continue;

    const w = image.naturalWidth || image.width;
    const h = image.naturalHeight || image.height;

    // Corners in local image coordinates (top-left origin)
    const corners = [
      [0, 0],
      [w, 0],
      [w, h],
      [0, h]
    ];

    // Transform each corner and update bounds
    for (const [x, y] of corners) {
      const [tx, ty] = applyTransform4x4(x, y, transform);
      minX = Math.min(minX, tx);
      minY = Math.min(minY, ty);
      maxX = Math.max(maxX, tx);
      maxY = Math.max(maxY, ty);
    }
  }

  if (minX === Infinity) return null;

  return { left: minX, top: minY, width: maxX - minX, height: maxY - minY };
}

function draw() {
  const imageSelector = '.original';
  background(220);

  // Don't animate until every image has been through the segmentation/alignment
  // pipeline and the playback schedule is ready — the raw #media images (now
  // visible in the page) show progress until then.
  if (playbackSchedule.length === 0) return;

  const scheduledIndices = playbackSchedule.map(e => e.index);
  mediaBoundingBox = getBoundingBox(imageSelector, scheduledIndices);

  const camPose = getCameraPose();
  if (camPose) {
    camera(
      camPose.eye[0], camPose.eye[1], camPose.eye[2],
      camPose.center[0], camPose.center[1], camPose.center[2],
      camPose.up[0], camPose.up[1], camPose.up[2]
    );
  }

  // Every image shows at a constant low alpha all the time (a translucent
  // overlapping "stack"), except whichever image's own moment is happening
  // right now — that one shows fully opaque, drawn last so it stands out
  // crisply on top rather than being dulled by another low-alpha overlay.
  // Depth writes are disabled throughout — otherwise the nearer quad's depth
  // value would block farther ones from blending through underneath it.
  const elapsed = getPlaybackElapsedMs();
  const momentHalfWindow = (MOMENT_MS * PLAYBACK_SPEED) / 2;

  const mediaElement = select('#media')?.elt;
  if(mediaElement) {
    push();
      drawingContext.depthMask(false);

      let momentEntry = null;

      for (let p = 0; p < playbackSchedule.length; p++) {
        const entry = playbackSchedule[p];
        const image = mediaElement.children[entry.index].querySelector(imageSelector);
        if (!image) continue;

        const inMoment = Math.abs(elapsed - entry.offsetMs) <= momentHalfWindow;
        if (inMoment && !momentEntry) momentEntry = entry;

        if (!inMoment) {
          push();
            tint(255, 255 * 0.2);
            const t = getImageTransformFromElement(image, true);
            drawProjectedImage(image, 0, 0, t, -p);
          pop();
        }
      }

      if (momentEntry) {
        const image = mediaElement.children[momentEntry.index].querySelector(imageSelector);
        if (image) {
          push();
            tint(255, 255);
            const t = getImageTransformFromElement(image, true);
            drawProjectedImage(image, 0, 0, t, 0);
          pop();
        }
      }

      drawingContext.depthMask(true);
    pop();
  }
}

function applyTransform4x4(px, py, M) {
  // strict: accept only flat row-major 4x4 arrays (length 16)
  if (!Array.isArray(M) || M.length !== 16) return [px, py];

  const X = M[0] * px + M[1] * py + M[2] * 0 + M[3];
  const Y = M[4] * px + M[5] * py + M[6] * 0 + M[7];
  const W = M[12] * px + M[13] * py + M[14] * 0 + M[15];

  if (!isFinite(W) || Math.abs(W) < 1e-12) return [X, Y];
  return [X / W, Y / W];
}

/**
 * Creates a new image element with the mask applied.
 * Pixels where the mask is dark (black) become transparent.
 * @param {HTMLImageElement|p5.Element} colorImg - the colour image
 * @param {HTMLImageElement|p5.Element} maskImg - the greyscale mask (white = keep, black = transparent)
 * @returns {p5.Element} - a new p5 img element containing the masked image
 */
function applyMaskToImage(colorImg, maskImg, invert = false, onloaded = () => {}) {
  let resultImg = createImg('', '');

  const w = colorImg.naturalWidth || colorImg.width;
  const h = colorImg.naturalHeight || colorImg.height;

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  ctx.drawImage(colorImg, 0, 0, w, h);

  const colorData = ctx.getImageData(0, 0, w, h);
  const cPixels = colorData.data;

  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(maskImg, 0, 0, w, h);
  const maskData = ctx.getImageData(0, 0, w, h);
  const mPixels = maskData.data;

  for (let i = 0; i < cPixels.length; i += 4) {
    const maskVal = invert ? 255 - mPixels[i] : mPixels[i];
    cPixels[i] = maskVal > 0 ? cPixels[i] : random(255);
    cPixels[i + 1] = maskVal > 0 ? cPixels[i + 1] : random(255);
    cPixels[i + 2] = maskVal > 0 ? cPixels[i + 2] : random(255);
    cPixels[i + 3] = maskVal;
  }

  ctx.putImageData(colorData, 0, 0);
  setImageTransform(resultImg.elt, getImageTransformFromElement(colorImg));

  resultImg.elt.onload = onloaded;
  resultImg.elt.src = canvas.toDataURL();

  return resultImg;
}

/**
 * Checks if a homography transform looks reasonable.
 * Returns { valid: boolean, reason: string, rotation: number, scale: number, shear: number }
 * 
 * A "reasonable" homography for image alignment should have:
 * - Minimal rotation (< maxRotationDeg)
 * - Scale close to 1 (within scaleRange)
 * - Low shear
 * - Low perspective distortion (bottom row close to [0, 0, 1])
 * 
 * @param {Array} H - flat 9-element row-major 3x3 homography, or flat 16-element 4x4
 * @param {Object} options - optional thresholds
 * @returns {Object} { valid, reason, rotation, scale, shear, perspective }
 */
function isReasonableHomography(H, options = {}) {
  const {
    maxRotationDeg = 15,      // max allowed rotation in degrees
    minScale = 0.5,           // min allowed scale
    maxScale = 2.0,           // max allowed scale
    maxShear = 0.3,           // max allowed shear
    maxPerspective = 0.001    // max allowed perspective distortion
  } = options;

  if (!H) return { valid: false, reason: 'H is null or undefined' };

  // extract 3x3 from flat 9 or flat 16
  let h00, h01, h02, h10, h11, h12, h20, h21, h22;
  if (H.length === 9) {
    [h00, h01, h02, h10, h11, h12, h20, h21, h22] = H;
  } else if (H.length === 16) {
    // 4x4 row-major: extract the 2D affine/projective part
    h00 = H[0];  h01 = H[1];  h02 = H[3];   // skip H[2] (z column)
    h10 = H[4];  h11 = H[5];  h12 = H[7];
    h20 = H[12]; h21 = H[13]; h22 = H[15];
  } else {
    return { valid: false, reason: 'H must be length 9 or 16' };
  }

  // normalize so h22 = 1 (if possible)
  if (Math.abs(h22) < 1e-12) {
    return { valid: false, reason: 'h22 is zero, degenerate homography' };
  }
  h00 /= h22; h01 /= h22; h02 /= h22;
  h10 /= h22; h11 /= h22; h12 /= h22;
  h20 /= h22; h21 /= h22; h22 = 1;

  // perspective distortion: bottom row should be [0, 0, 1]
  const perspective = Math.sqrt(h20 * h20 + h21 * h21);
  if (perspective > maxPerspective) {
    return {
      valid: false,
      reason: `Perspective distortion too high: ${perspective.toFixed(6)} > ${maxPerspective}`,
      perspective
    };
  }

  // decompose upper-left 2x2 into rotation, scale, shear
  // H = [ a  b  tx ]   where [a b; c d] = R * S * Shear
  //     [ c  d  ty ]
  //     [ 0  0  1  ]
  const a = h00, b = h01, c = h10, d = h11;

  // scale: sqrt of determinant gives overall scale
  const det = a * d - b * c;
  if (det <= 0) {
    return { valid: false, reason: 'Negative or zero determinant (flipped or degenerate)' };
  }
  const scale = Math.sqrt(det);

  // rotation angle from the 2x2 matrix (assumes no/low shear)
  // rotation = atan2(c, a) for a proper rotation matrix
  const rotationRad = Math.atan2(c, a);
  const rotationDeg = Math.abs(rotationRad * 180 / Math.PI);

  // shear: measure how non-orthogonal the axes are
  // shear ~ (a*b + c*d) / det for normalized matrix
  const shear = Math.abs(a * b + c * d) / det;

  // check thresholds
  if (rotationDeg > maxRotationDeg) {
    return {
      valid: false,
      reason: `Rotation too large: ${rotationDeg.toFixed(2)}° > ${maxRotationDeg}°`,
      rotation: rotationDeg,
      scale,
      shear,
      perspective
    };
  }

  if (scale < minScale || scale > maxScale) {
    return {
      valid: false,
      reason: `Scale out of range: ${scale.toFixed(3)} not in [${minScale}, ${maxScale}]`,
      rotation: rotationDeg,
      scale,
      shear,
      perspective
    };
  }

  if (shear > maxShear) {
    return {
      valid: false,
      reason: `Shear too high: ${shear.toFixed(3)} > ${maxShear}`,
      rotation: rotationDeg,
      scale,
      shear,
      perspective
    };
  }

  return {
    valid: true,
    reason: 'OK',
    rotation: rotationDeg,
    scale,
    shear,
    perspective
  };
}

/**
 * Multiplies two 4x4 row-major flat matrices and returns the result.
 * Result = A * B (A applied first, then B)
 * @param {Array} A - flat 16-element row-major 4x4 matrix
 * @param {Array} B - flat 16-element row-major 4x4 matrix
 * @returns {Array} - flat 16-element row-major 4x4 matrix (A * B)
 */
function multiplyMatrix4x4(A, B) {
  let result = null;

  if (!A || A.length !== 16 || !B || B.length !== 16) {
  }
  else {
    result = new Array(16);

    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
          sum += A[row * 4 + k] * B[k * 4 + col];
        }
        result[row * 4 + col] = sum;
      }
    }
  }

  return result;
}

function invertMatrix4x4(A) {
  const inv = new Array(16);
  const det = determinant4x4(A);
  if (det === 0) {
    return null;
  }
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      inv[j * 4 + i] = cofactor4x4(A, i, j) / det;
    }
  }
  return inv;
}

function determinant4x4(m) {
  if (!m || m.length !== 16) return null;

  // Helper for 3x3 determinant
  function det3(a, b, c, d, e, f, g, h, i) {
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  }

  const m0 = m[0],  m1 = m[1],  m2 = m[2],  m3 = m[3],
        m4 = m[4],  m5 = m[5],  m6 = m[6],  m7 = m[7],
        m8 = m[8],  m9 = m[9],  m10 = m[10], m11 = m[11],
        m12 = m[12], m13 = m[13], m14 = m[14], m15 = m[15];

  return (
    m0 * det3(m5, m6, m7,  m9, m10, m11,  m13, m14, m15)
    - m1 * det3(m4, m6, m7,  m8, m10, m11,  m12, m14, m15)
    + m2 * det3(m4, m5, m7,  m8, m9, m11,  m12, m13, m15)
    - m3 * det3(m4, m5, m6,  m8, m9, m10,  m12, m13, m14)
  );
}

function cofactor4x4(m, row, col) {
  // Build the 3x3 minor by skipping the given row and column
  const minor = [];
  for (let i = 0; i < 4; i++) {
    if (i === row) continue;
    for (let j = 0; j < 4; j++) {
      if (j === col) continue;
      minor.push(m[i * 4 + j]);
    }
  }
  // Compute the determinant of the 3x3 minor
  const det =
    minor[0] * (minor[4] * minor[8] - minor[5] * minor[7]) -
    minor[1] * (minor[3] * minor[8] - minor[5] * minor[6]) +
    minor[2] * (minor[3] * minor[7] - minor[4] * minor[6]);
  // Apply the checkerboard sign
  return ((row + col) % 2 === 0 ? 1 : -1) * det;
}

function keyPressed() {
  if (key === 'x' || key === 'X') {
    exportAllMediaElements('.original');
  }
}

/**
 * Creates as many fully transparent PNG image elements as there are media elements,
 * each with the dimensions of mediaBoundingBox, draws the corresponding media image into it
 * using its transform relative to the bounding box, and appends them to #export.
 */
function exportAllMediaElements(selector) {
  const mediaElement = select('#media')?.elt;
  const exportElement = select('#export')?.elt;
  if (!mediaElement || !exportElement || !mediaBoundingBox) {
    console.warn('Missing #media, #export, or mediaBoundingBox');
    return;
  }

  // Clear previous exports
  exportElement.innerHTML = '';

  const w = Math.round(mediaBoundingBox.width);
  const h = Math.round(mediaBoundingBox.height);

  for (let i = 0; i < mediaElement.children.length; i++) {
    // Create a canvas for export
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    // Fill with fully transparent background
    ctx.clearRect(0, 0, w, h);

    // Draw the media image (e.g. .lowres) into the canvas, using its transform
    const img = mediaElement.children[i].querySelector(selector);
    const transform = getImageTransformFromElement(img, true);
    console.log(transform);
    if (img && transform) {
      const imgW = img.naturalWidth || img.width;
      const imgH = img.naturalHeight || img.height;

      // Extract 2D affine for canvas setTransform(a, b, c, d, e, f)
      // Canvas: | a c e |  Our 4x4: X = M[0]*x + M[1]*y + M[3]
      //         | b d f |           Y = M[4]*x + M[5]*y + M[7]
      const a = transform[0];
      const b = transform[4];
      const c = transform[1];
      const d = transform[5];
      const tx = transform[3] - mediaBoundingBox.left;
      const ty = transform[7] - mediaBoundingBox.top;

      ctx.save();
      ctx.setTransform(a, b, c, d, tx, ty);
      ctx.drawImage(img, 0, 0, imgW, imgH);
      ctx.restore();
    }

    // Create an image element from the canvas
    const outImg = document.createElement('img');
    outImg.src = canvas.toDataURL('image/png');
    outImg.width = w;
    outImg.height = h;
    outImg.style.width = w + "px";
    outImg.style.height = h + "px";

    exportElement.appendChild(outImg);
  }
}

// Orders two images by their recovered capture sequence: EXIF timestamp when
// both have one, otherwise alphabetically by filename (images with a
// timestamp always sort before ones without).
function compareImagesForSequence(imgA, imgB) {
  const ta = getImageTimestampFromElement(imgA);
  const tb = getImageTimestampFromElement(imgB);
  if (ta && tb) return ta.getTime() - tb.getTime();
  if (ta) return -1;
  if (tb) return 1;
  return imgA.src.localeCompare(imgB.src);
}

async function processAnyAttachedMedia() {
  const originals = selectAll('#media .original');
  // Wait for all images to load
  await Promise.all(originals.map(i => {
    return new Promise(resolve => {
      if (i.elt.complete) resolve();
      else {
        i.elt.onload = resolve;
        i.elt.onerror = resolve;
      }
    });
  }));

  // Phase 1: recover every image's EXIF timestamp up front, before any
  // alignment work — the processing order below (and buildPlaybackSchedule
  // afterwards) is derived from these.
  for (const orig of originals) {
    setImageTransform(orig.elt, identityMatrix);
    const timestamp = await extractImageTimestamp(orig.elt);
    setImageTimestamp(orig.elt, timestamp);
  }

  // Phase 2: segment + align in chronological (or alphabetical-fallback)
  // order, so each new image is matched against the nearest one actually
  // preceding it in the recovered sequence.
  const ordered = [...originals].sort((a, b) => compareImagesForSequence(a.elt, b.elt));
  for (const orig of ordered) {
    await processImage(orig.elt, orig.parent());
    processHomography(orig.parent().id);
  }

  buildPlaybackSchedule();
  buildCameraKeyframes();
}

// Orders images by their recovered capture sequence and records each one's
// offset (in ms) from the first frame, so draw()/the camera can play the
// sequence back at the same pace it was actually shot.
function buildPlaybackSchedule() {
  const mediaElement = select('#media')?.elt;
  if (!mediaElement) return;

  const entries = [];
  for (let i = 0; i < mediaElement.children.length; i++) {
    const image = mediaElement.children[i].querySelector('.original');
    if (image) entries.push({ index: i, div: mediaElement.children[i], image });
  }
  if (entries.length === 0) return;

  entries.sort((a, b) => compareImagesForSequence(a.image, b.image));

  // Stop the sequence at the first frame with no valid alignment transform,
  // rather than including a mispositioned/unaligned frame in playback.
  const aligned = [];
  for (const e of entries) {
    if (!getImageTransformFromElement(e.div)) break;
    aligned.push(e);
  }
  if (aligned.length === 0) return;

  const t0 = getImageTimestampFromElement(aligned[0].image)?.getTime();

  let lastOffset = -FALLBACK_FRAME_SPACING_MS;
  playbackSchedule = aligned.map(e => {
    const t = getImageTimestampFromElement(e.image)?.getTime();
    const offsetMs = (t !== undefined && t0 !== undefined) ? (t - t0) : (lastOffset + FALLBACK_FRAME_SPACING_MS);
    lastOffset = offsetMs;
    return { index: e.index, offsetMs };
  });
  playbackStartMillis = millis();
}

// Elapsed time (ms) within the current looping playback cycle, in the same
// units as playbackSchedule's offsetMs — shared by image-frame selection and
// camera keyframe animation so the two always stay in lockstep. Only the travel
// between the first and last keyframe is scaled by PLAYBACK_SPEED; the
// end-of-sequence pause (PLAYBACK_END_PAUSE_MS) always lasts that long in real
// wall-clock time, regardless of speed. Elapsed keeps advancing past the last
// keyframe's time during the pause (rather than holding there), so the last
// image's moment window closes just like every other frame's — leaving the
// screen blank (camera holds its final position) until the sequence loops.
function getPlaybackElapsedMs() {
  if (playbackSchedule.length === 0) return 0;

  const lastOffset = playbackSchedule[playbackSchedule.length - 1].offsetMs;
  const realTravelDuration = lastOffset / PLAYBACK_SPEED;
  const realCycleLength = realTravelDuration + PLAYBACK_END_PAUSE_MS;
  const realElapsed = (millis() - playbackStartMillis) % realCycleLength;

  return realElapsed * PLAYBACK_SPEED;
}

// Fits a single image edge-to-edge ("square" to the camera) into a
// perspective camera's view, deriving size/center/roll directly from its own
// transformed corners (via the same applyTransform4x4 used to draw it) —
// an aligned image is often rotated slightly in world space, so an
// axis-aligned bounding box around it (viewed by a non-rolled camera) would
// leave gaps rather than exactly filling the frame.
function computeCameraKeyframeForImage(image, transform) {
  const w = image.naturalWidth || image.width;
  const h = image.naturalHeight || image.height;

  const [cx, cy] = applyTransform4x4(w / 2, h / 2, transform);
  const [tlx, tly] = applyTransform4x4(0, 0, transform);
  const [trx, tryy] = applyTransform4x4(w, 0, transform);
  const [blx, bly] = applyTransform4x4(0, h, transform);

  // Right/up edge vectors of the transformed image in world space — their
  // lengths give the image's true (un-inflated) footprint, and the up
  // vector's direction gives the camera roll needed to match the image's own
  // rotation, so the frame fills edge-to-edge instead of leaving gaps.
  const worldW = Math.hypot(trx - tlx, tryy - tly);
  const worldH = Math.hypot(blx - tlx, bly - tly);
  const upLen = Math.hypot(blx - tlx, bly - tly) || 1;
  const up = [(blx - tlx) / upLen, (bly - tly) / upLen, 0];

  const fovY = PI / 3; // p5's default WEBGL vertical field of view (60deg)
  const aspect = width / height;

  const distForHeight = (worldH / 2) / Math.tan(fovY / 2);
  const distForWidth = (worldW / 2) / (Math.tan(fovY / 2) * aspect);
  const dist = Math.max(distForHeight, distForWidth);

  return {
    eye: [cx, cy, dist],
    center: [cx, cy, 0],
    up
  };
}

// Derives one camera keyframe per successfully-aligned image (from the same
// transforms buildPlaybackSchedule just validated), framing that image alone.
// Each keyframe's time matches that image's playbackSchedule offset exactly,
// so the camera is precisely, squarely framed on an image the moment it's due.
function buildCameraKeyframes() {
  const mediaElement = select('#media')?.elt;
  if (!mediaElement || playbackSchedule.length === 0) { cameraKeyframes = []; return; }

  cameraKeyframes = playbackSchedule.map(entry => {
    const image = mediaElement.children[entry.index].querySelector('.original');
    const transform = getImageTransformFromElement(image, true);
    const pose = computeCameraKeyframeForImage(image, transform);
    return { time: entry.offsetMs, ...pose };
  });
}

function lerp3(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t
  ];
}

// Returns the interpolated {eye, center, up} camera pose for the current
// moment: exactly kfA at kfA.time, exactly kfB at kfB.time, travelling smoothly
// between — using the same elapsed clock that gates each image's own moment
// window in draw(), so the camera is always centred on whichever image (if
// any) is currently in its moment.
function getCameraPose() {
  if (cameraKeyframes.length === 0) return null;
  if (cameraKeyframes.length === 1) return cameraKeyframes[0];

  const elapsed = getPlaybackElapsedMs();

  let i = 0;
  while (i < cameraKeyframes.length - 1 && cameraKeyframes[i + 1].time <= elapsed) i++;

  const kfA = cameraKeyframes[i];
  const kfB = cameraKeyframes[Math.min(i + 1, cameraKeyframes.length - 1)];
  if (kfA === kfB) return kfA;

  const span = kfB.time - kfA.time;
  const t = span > 0 ? constrain((elapsed - kfA.time) / span, 0, 1) : 1;

  return {
    eye: lerp3(kfA.eye, kfB.eye, t),
    center: lerp3(kfA.center, kfB.center, t),
    up: lerp3(kfA.up, kfB.up, t)
  };
}

// Reads DateTimeOriginal + SubSecTimeOriginal via exif-js and combines them
// into a single Date with millisecond precision (EXIF only stores whole seconds).
function extractImageTimestamp(imgElement) {
  return new Promise((resolve) => {
    EXIF.getData(imgElement, function () {
      const dateTimeOriginal = EXIF.getTag(this, 'DateTimeOriginal');
      const subSecTimeOriginal = EXIF.getTag(this, 'SubsecTimeOriginal');
      const timestamp = parseExifDateTime(dateTimeOriginal, subSecTimeOriginal);
      console.log('EXIF timestamp for', imgElement.src, ':', dateTimeOriginal, subSecTimeOriginal, '->', timestamp);
      resolve(timestamp);
    });
  });
}

function parseExifDateTime(dateTimeOriginal, subSecTimeOriginal) {
  if (!dateTimeOriginal) return null;

  const match = dateTimeOriginal.match(/^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})$/);
  if (!match) return null;

  const [year, month, day, hour, minute, second] = match.slice(1).map(Number);
  const ms = subSecTimeOriginal ? Math.round(parseFloat('0.' + subSecTimeOriginal) * 1000) : 0;

  // EXIF months are 1-indexed; JS Date months are 0-indexed
  return new Date(year, month - 1, day, hour, minute, second, ms);
}

function setImageTimestamp(element, timestamp) {
  console.log('setImageTimestamp', element, timestamp);
  if (element && timestamp instanceof Date) {
    element.setAttribute('data-timestamp', timestamp.getTime());
  }
}

function getImageTimestampFromElement(element) {
  if (element) {
    const t = element.getAttribute('data-timestamp');
    if (t !== null) return new Date(Number(t));
  }
  return null;
}

async function processImage(originalImgElement, div) {
  // 1. Generate low-res image and wait for it to load
  const lowResImg = await generateLowResImageAsync(originalImgElement);
  lowResImg.parent(div);
  lowResImg.addClass('lowres');

  // 2. Generate mask and wait for it to load
  const maskImg = await generateMaskAsync(lowResImg.elt);
  maskImg.parent(div);
  maskImg.addClass('mask');

  // 3. Apply mask to get foreground and background, wait for both
  const [foregroundImg, backgroundImg] = await Promise.all([
    applyMaskToImageAsync(lowResImg.elt, maskImg.elt, false),
    applyMaskToImageAsync(lowResImg.elt, maskImg.elt, true)
  ]);
  foregroundImg.parent(div);
  foregroundImg.addClass('foreground');
  backgroundImg.parent(div);
  backgroundImg.addClass('background');
}

// Helper: Promise version of generateLowResImage
function generateLowResImageAsync(imgElement) {
  return new Promise(resolve => {
    const lowresImg = generateLowResImage(imgElement, () => resolve(lowresImg));
  });
}

// Helper: Promise version of generateMask
function generateMaskAsync(imgElement) {
  return new Promise(resolve => {
    const maskImg = generateMask(imgElement, () => resolve(maskImg));
  });
}

// Helper: Promise version of applyMaskToImage
function applyMaskToImageAsync(colorImg, maskImg, invert) {
  return new Promise(resolve => {
    const resultImg = applyMaskToImage(colorImg, maskImg, invert, () => resolve(resultImg));
  });
}

function setImageTransform(element, transform) {
  console.log('setImageTransform', element, transform);
  if (element && Array.isArray(transform)) {
    element.setAttribute('data-transform', JSON.stringify(transform));
  }
}

function getImageTransformFromElement(element, traverse = false) {
  let result = null;

  if (element){
    const b = traverse ? (getImageTransformFromElement(element.parentElement, false) || identityMatrix) : identityMatrix;
    try {
      result = JSON.parse(element.getAttribute('data-transform'));
    }
    catch (e) {
    }
    if(result) result = multiplyMatrix4x4(b, result);
  }

  return result;
}