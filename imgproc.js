// imgproc.js - generic image alignment helpers for opencv-featurematch
// Extracted from the segmention branch: matrix math, homography validation/cleanup,
// and a clean two-image alignment primitive (alignImagePair). Deliberately just
// the math - no DOM/canvas conventions (how a transform gets stored on an
// element, how images get downscaled or masked, how a texture gets rendered)
// and no EXIF, camera, playback, or multi-image search logic. All of that is
// application-specific and stays in each app's own sketch.js.
//
// Depends on align_img.js being loaded first (uses its globals: Align_img, h,
// good_inlier_matches).

const identityMatrix = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
];

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
 * Multiplies two 4x4 row-major flat matrices and returns the literal matrix
 * product A * B. applyTransform4x4 applies a matrix to a column vector
 * (M * p), so to compose "apply A to a point first, then apply B to the
 * result" - i.e. B * (A * p) - call multiplyMatrix4x4(B, A), not (A, B).
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

// Strips shear out of a 4x4 transform, rebuilding it as a pure rotation + uniform
// scale from its top-row (a, c) components. Real (especially weakly-matched)
// homographies carry small shear, which corrupts anything that derives rotation
// from more than one edge of the transformed image (e.g. camera framing that uses
// a left/bottom edge as well as the top edge). Deriving from a single edge keeps
// callers immune to that noise.
function stripShear(transform) {
  if (!transform) return transform;

  const a = transform[0], c = transform[4];
  const tx = transform[3], ty = transform[7];

  const scale = Math.hypot(a, c) || 1;
  const cosT = a / scale, sinT = c / scale;

  return [
    scale * cosT, -scale * sinT, 0, tx,
    scale * sinT,  scale * cosT, 0, ty,
    0, 0, 1, 0,
    0, 0, 0, 1
  ];
}

/**
 * Aligns two images and returns the result directly, instead of forcing the
 * caller to read Align_img's side-effect globals (h, good_inlier_matches).
 * Synchronous - every OpenCV.js call inside Align_img (detectAndCompute,
 * knnMatch, findHomography) is a synchronous WASM operation, nothing here
 * is ever awaited. Align_img throws (rather than alert()ing) if it can't
 * find enough matches or a homography - caught here and folded into the
 * same { valid: false, reason } shape as every other failure case, so
 * callers only ever need to check .valid, never wrap this in a try/catch.
 * @param {HTMLImageElement} imageA - reference image
 * @param {HTMLImageElement} imageB - image to align onto imageA
 * @param {Object} options - passed through to isReasonableHomography
 * @returns {{valid: boolean, transform: (Array|null), inliers: number, reason: string}}
 */
function alignImagePair(imageA, imageB, options = {}) {
  if (!imageA || !imageB) {
    return { valid: false, transform: null, inliers: 0, reason: 'imageA or imageB is null or undefined' };
  }

  try {
    Align_img(imageA, imageB);
  } catch (err) {
    return { valid: false, transform: null, inliers: 0, reason: err.message };
  }

  const inliers = (good_inlier_matches && good_inlier_matches.size) ? good_inlier_matches.size() : 0;

  if (!h || h.empty() || !h.data64F) {
    return { valid: false, transform: null, inliers, reason: 'No homography found' };
  }

  const check = isReasonableHomography(Array.from(h.data64F), options);
  if (!check.valid) {
    return { valid: false, transform: null, inliers, reason: check.reason };
  }

  const transform = [
    h.data64F[0], h.data64F[1], 0, h.data64F[2],
    h.data64F[3], h.data64F[4], 0, h.data64F[5],
    0, 0, 1, 0,
    h.data64F[6], h.data64F[7], 0, h.data64F[8]
  ];

  return { valid: true, transform, inliers, reason: check.reason };
}
