/*!
 * opencv-featurematch-js.js - feature-based image alignment for OpenCV.js
 *
 * Combines two parts that used to ship as separate files:
 *   1. Align_img() - the core feature-matching/homography computation.
 *      Originally adapted from Scott Suhy's tutorial "Image Alignment
 *      (Feature Based) in OpenCV.js":
 *      https://web.archive.org/web/20210201184709/https://scottsuhy.com/2021/02/01/image-alignment-feature-based-in-opencv-js-javascript/
 *      (original link now dead; credit kept below alongside the code it applies to).
 *   2. imgproc - pure alignment math on top: 4x4 matrix helpers, homography
 *      validation/cleanup, and alignImagePair(), the clean two-image
 *      primitive most consumers actually want.
 *
 * No DOM/canvas conventions, no rendering, no EXIF/camera/playback, and no
 * multi-image sequencing policy - all of that is application-specific and
 * stays in each app's own sketch.js. davidchatting/shimage covers the
 * p5.js/WEBGL rendering side.
 */

// opencv.js's own <script onload> fires once its JS wrapper has loaded, not
// once its WASM runtime has actually finished initializing - calling
// anything here (Align_img, alignImagePair) before that finishes throws
// "undefined is not a constructor". cv.Mat is a reliable proxy for "the
// runtime is ready": in practice it becomes available at the same instant as
// every other bound class. cv is technically thenable, but awaiting it
// directly never resolves - use this instead. Callers should `await
// cvLoaded()` once before their first call into this library.
function cvLoaded() {
  return new Promise(resolve => {
    if (typeof cv !== 'undefined' && cv.Mat) {
      resolve();
      return;
    }
    // cv may not exist as a global at all yet (opencv.js script tag still
    // loading/executing) - only attach onRuntimeInitialized once it does.
    if (typeof cv !== 'undefined') {
      cv['onRuntimeInitialized'] = resolve;
    }
    // Fallback poll, in case cv doesn't exist yet above, or
    // onRuntimeInitialized already fired in the race between the check
    // above and this assignment.
    (function poll() {
      if (typeof cv !== 'undefined' && cv.Mat) resolve();
      else setTimeout(poll, 50);
    })();
  });
}

var canvas;
var inputImageA = null, inputImageB = null;
var points1 = [];
var points2 = [];
var good_inlier_matches;
var h;
var good_matches_global = null; // store matches so draw() can render them

function Align_img(image_element_a, image_element_b) {
   if (!image_element_a || !image_element_b) return null;

   //Based on: https://web.archive.org/web/20210201184709/https://scottsuhy.com/2021/02/01/image-alignment-feature-based-in-opencv-js-javascript/ (original now dead)
   // reset previous state so repeated presses don't append results
   points1 = [];
   points2 = [];
   good_inlier_matches = new cv.DMatchVector();
   good_matches_global = null;

  let detector_option = 2;  //KAZE document.getElementById('detector').value;
  let match_option = 1;  //knnMatch document.getElementById('match').value;
  let matchDistance_option = 20;  //document.getElementById('distance').value;
  let knnDistance_option = 0.7;  //document.getElementById('knn_distance').value;
  let pyrDown_option = "No";  //document.getElementById('pyrDown').value;

  console.error("STEP 1: READ IN IMAGES **********************************************************************");
  //im2 is the original reference image we are trying to align to
  let im2 = cv.imread(image_element_a);
  getMatStats(im2, "original reference image");
  //im1 is the image we are trying to line up correctly
  
  let resultSize = im2.size();
  // inputImageA/inputImageB are only used by a consuming sketch's optional
  // debug preview (e.g. drawMatchesOverlay) - createImage is a p5.js global,
  // so this is skipped entirely when p5.js isn't loaded (a plain-DOM consumer
  // calling this for its transform/inlier result has no use for them anyway).
  if (typeof createImage === 'function') {
    inputImageB = createImage(resultSize.width, resultSize.height);
    cvMatToP5Image(im2, inputImageB);
  }

  let im1 = cv.imread(image_element_b);

  if (typeof createImage === 'function') {
    resultSize = im1.size();
    inputImageA = createImage(resultSize.width, resultSize.height);
    cvMatToP5Image(im1, inputImageA);
  }

  getMatStats(im1, "original image to line up");

  if (pyrDown_option !== 'No') {
      console.log("User selected option to pyrDown image");
      cv.pyrDown(im1, im1, new cv.Size(0, 0), cv.BORDER_DEFAULT);
      cv.pyrDown(im2, im2, new cv.Size(0, 0), cv.BORDER_DEFAULT);
      getMatStats(im1, "new stats for im1");
      getMatStats(im2, "new stats for im2");
  }

  console.error("STEP 2: CONVERT IMAGES TO GRAYSCALE *********************************************************");
  //17            Convert images to grayscale
  //18            Mat im1Gray, im2Gray;
  //19            cvtColor(im1, im1Gray, CV_BGR2GRAY);
  //20            cvtColor(im2, im2Gray, CV_BGR2GRAY);
  let im1Gray = new cv.Mat();
  let im2Gray = new cv.Mat();
  cv.cvtColor(im1, im1Gray, cv.COLOR_BGRA2GRAY);
  getMatStats(im1Gray, "reference image converted to BGRA2GRAY");
  cv.cvtColor(im2, im2Gray, cv.COLOR_BGRA2GRAY);
  getMatStats(im2Gray, "image to line up converted to BGRA2GRAY");

  console.error("STEP 3: DETECT FEATURES & COMPUTE DESCRIPTORS************************************************");
  //22            Variables to store keypoints and descriptors
  //23            std::vector<KeyPoint> keypoints1, keypoints2;
  //24            Mat descriptors1, descriptors2;
  let keypoints1 = new cv.KeyPointVector();
  let keypoints2 = new cv.KeyPointVector();
  let descriptors1 = new cv.Mat();
  let descriptors2 = new cv.Mat();
  //26            Detect ORB features and compute descriptors.
  //27            Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
  //28            orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
  //29            orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

  if (detector_option == 0) {
      var X = new cv.ORB(5000);
      console.log("using cv.ORB");
  } else if (detector_option == 1) {
      var X = new cv.AKAZE();
      console.log("using cv.AKAZE");
  } else if (detector_option == 2) {
      var X = new cv.KAZE();
      console.log("using cv.KAZE");
  }

  X.detectAndCompute(im1Gray, new cv.Mat(), keypoints1, descriptors1);
  X.detectAndCompute(im2Gray, new cv.Mat(), keypoints2, descriptors2);

  console.log("keypoints1: ", keypoints1);
  console.log("descriptors1: ", descriptors1);
  console.log("keypoints2: ", keypoints2);
  console.log("descriptors2: ", descriptors2);
  getMatStats(descriptors1, "descriptors1");
  getMatStats(descriptors2, "descriptors2");

  // use to debug and list out all the keypoints
  console.log("there are a total of ", keypoints1.size(), " keypoints1 (img to aligned) and ", keypoints2.size(), " keypoints2 (reference)");
  console.log("here are the first 5 keypoints for keypoints1 - image to align.");
  for (let i = 0; i < keypoints1.size(); i++) {
      console.log("keypoints1: [",i,"]", keypoints1.get(i).pt.x, keypoints1.get(i).pt.y);
      if (i === 5){break;}
  }

  console.log("here are the first 5 keypoints for keypoints2 -- reference image");
  for (let i = 0; i < keypoints2.size(); i++) {
      console.log("keypoints2: [",i,"]", keypoints2.get(i).pt.x, keypoints2.get(i).pt.y);
      if (i === 5){break;}
  }

  console.log("there are a total of [", descriptors1.cols, "][", descriptors1.rows, "] descriptors1 [cols][rows] (img to aligned) and [", descriptors2.cols, "][", descriptors2.rows, "] descriptors2 (reference) [cols][rows]");

  console.error("STEP 4: MATCH FEATURES **********************************************************************");
  //31            Match features.
  //32            std::vector<DMatch> matches;
  //33            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //34            matcher->match(descriptors1, descriptors2, matches, Mat());

  let good_matches = new cv.DMatchVector();
  // expose matches to drawing routine
  good_matches_global = good_matches;

  if(match_option == 0){//match
      console.log("using match...");
      let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
      let matches = new cv.DMatchVector();
      bf.match(descriptors1, descriptors2, matches);

      //36            Sort matches by score
      //37            std::sort(matches.begin(), matches.end());
      //39            Remove not so good matches
      //40            const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
      //41            matches.erase(matches.begin()+numGoodMatches, matches.end());
      console.log("matches.size: ", matches.size());
      for (let i = 0; i < matches.size(); i++) {
          if (matches.get(i).distance < matchDistance_option) {
              good_matches.push_back(matches.get(i));
          }
      }
      if(good_matches.size() <= 3){
          throw new Error("Less than 4 good matches found! counter =" + good_matches.size() + " try changing distance.");
      }
  }
  else if(match_option == 1) { //knnMatch
      console.log("using knnMatch...");
      let bf = new cv.BFMatcher();
      let matches = new cv.DMatchVectorVector();
      //Reference: https://docs.opencv.org/3.3.0/db/d39/classcv_1_1DescriptorMatcher.html#a378f35c9b1a5dfa4022839a45cdf0e89
      bf.knnMatch(descriptors1, descriptors2, matches, 2);

      let counter = 0;
      for (let i = 0; i < matches.size(); ++i) {
          let match = matches.get(i);
          let dMatch1 = match.get(0);
          let dMatch2 = match.get(1);
          //console.log("[", i, "] ", "dMatch1: ", dMatch1, "dMatch2: ", dMatch2);
          if (dMatch1.distance <= dMatch2.distance * parseFloat(knnDistance_option)) {
              //console.log("***Good Match***", "dMatch1.distance: ", dMatch1.distance, "was less than or = to: ", "dMatch2.distance * parseFloat(knnDistance_option)", dMatch2.distance * parseFloat(knnDistance_option), "dMatch2.distance: ", dMatch2.distance, "knnDistance", knnDistance_option);
              good_matches.push_back(dMatch1);
              counter++;
          }
      }
      if(counter <= 3){
          throw new Error("Less than 4 good matches found! Counter=" + counter + " try changing distance %. It's currently " + knnDistance_option);
      }
      console.log("keeping ", counter, " points in good_matches vector out of ", matches.size(), " contained in this match vector:", matches);
      console.log("here are first 5 matches");
      for (let t = 0; t < matches.size(); ++t) {
          console.log("[" + t + "]", "matches: ", matches.get(t));
          if (t === 5){break;}
      }
  }
  console.log("here are first 5 good_matches");
  for (let r = 0; r < good_matches.size(); ++r) {
      console.log("[" + r + "]", "good_matches: ", good_matches.get(r));
      if (r === 5){break;}
  }

  console.error("STEP 5: DRAW TOP MATCHES AND OUTPUT IMAGE TO SCREEN ***************************************");
  //44            Draw top matches
  //45            Mat imMatches;
  //46            drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
  //47            imwrite("matches.jpg", imMatches);
  let imMatches = new cv.Mat();
  let color = new cv.Scalar(0,255,0, 255);
  //cv.drawMatches(im1, keypoints1, im2, keypoints2, good_matches, imMatches, color);
  //cv.imshow('imageCompareMatches', imMatches);
  getMatStats(imMatches, "imMatches");

  console.error("STEP 6: EXTRACT LOCATION OF GOOD MATCHES AND BUILD POINT1 and POINT2 ARRAYS ***************");
  //50            Extract location of good matches
  //51            std::vector<Point2f> points1, points2;
  //53            for( size_t i = 0; i < matches.size(); i++ )
  //54            {
  //55                points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
  //56                points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
  //57            }

  //this is a test
  //let points1 = create2dPointsArray(good_matches.size(), 2, 0);
  //let points2 = create2dPointsArray(good_matches.size(), 2, 0);
  
  for (let i = 0; i < good_matches.size(); i++) {

      points1.push(keypoints1.get(good_matches.get(i).queryIdx ).pt.x );
      points1.push(keypoints1.get(good_matches.get(i).queryIdx ).pt.y );
      points2.push(keypoints2.get(good_matches.get(i).trainIdx ).pt.x );
      points2.push(keypoints2.get(good_matches.get(i).trainIdx ).pt.y );
  }
  console.log("points1:", points1,"points2:", points2);

  console.error("STEP 7: CREATE MAT1 and MAT2 FROM POINT1 and POINT2 ARRAYS ********************************");
  //Alternative:
  //let mat1 = cv.matFromArray(points1.length, 1, cv.CV_32FC2, points1);
  //let mat2 = cv.matFromArray(points2.length, 1, cv.CV_32FC2, points2);

  // Create mats with one row per MATCH (not one row per float).
  // number of matches == good_matches.size()
  const numMatches = good_matches.size();
  if (numMatches === 0) {
    console.error("No matches found, aborting homography step.");
    return;
  }
  // matFromArray expects (rows, cols, type, array)
  let mat1 = cv.matFromArray(numMatches, 1, cv.CV_32FC2, points1);
  let mat2 = cv.matFromArray(numMatches, 1, cv.CV_32FC2, points2);
 
   getMatStats(mat1, "mat1 prior to homography");
   getMatStats(mat2, "mat2 prior to homography");
 
   console.error("STEP 8: CALCULATE HOMOGRAPHY USING MAT1 and MAT2 ******************************************");
  //59            Find homography
  //60            h = findHomography( points1, points2, RANSAC );
  //Reference: https://docs.opencv.org/3.3.0/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
  //mat1:	Coordinates of the points in the original plane, a matrix of the type CV_32FC2 or vector<Point2f> .
  //mat2:	Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or a vector<Point2f> .

  let findHomographyMask = new cv.Mat();
  h = cv.findHomography(mat1, mat2, cv.RANSAC, 3, findHomographyMask);
   if (h.empty())
   {
       throw new Error("homography matrix empty!");
   }
   else{
      console.log("h:", h);
      console.log("[", h.data64F[0],",", h.data64F[1], ",",h.data64F[2]);
      console.log("", h.data64F[3],",", h.data64F[4], ",", h.data64F[5]);
      console.log("", h.data64F[6],",", h.data64F[7], ",", h.data64F[8], "]");

      getMatStats(findHomographyMask, "findHomographyMask"); //test
      console.log("here are the inliers from RANSAC, compare to the good_matches array above", findHomographyMask.rows);//test
      good_inlier_matches = new cv.DMatchVector();
      for (let i = 0; i < findHomographyMask.rows; ++i) {
          if (findHomographyMask.data[i] === 1) {
              // the i-th mask entry corresponds to the i-th match in good_matches
              good_inlier_matches.push_back(good_matches.get(i));
          }
      }
      var inlierMatches = new cv.Mat();
      console.log("Good Matches: ", good_matches.size(), " inlier Matches: ", good_inlier_matches.size());

      console.log("here are inlier good_matches");
      for (let r = 0; r < good_inlier_matches.size(); ++r) {
          console.log("[" + r + "]", "good_inlier_matches: ", good_inlier_matches.get(r));
          //console.log(keypoints1[good_inlier_matches.get(r).queryIdx], keypoints2[good_inlier_matches.get(r).trainIdx]);
      }
      
      let src = cv.matFromArray(3, 1, cv.CV_32FC2, [0,0,1]);
      getMatStats(src, "src");
  }
  getMatStats(findHomographyMask, "findHomographyMask");
  // free mask now that we used it (keep good_inlier_matches)
  // (if you still need findHomographyMask for debug then remove this delete)
  findHomographyMask.delete();

  console.error("STEP 9: WARP IMAGE TO ALIGN WITH REFERENCE **************************************************");
  //62          Use homography to warp image
  //63          warpPerspective(im1, im1Reg, h, im2.size());
  //Reference: https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
  let image_B_final_result = new cv.Mat();
  cv.warpPerspective(im1, image_B_final_result, h, im2.size());
  //cv.imshow('image_Aligned', image_B_final_result);
  getMatStats(image_B_final_result, "finalMat");

  //X.delete();
  descriptors1.delete();
  descriptors2.delete();
  keypoints1.delete();
  keypoints2.delete();
  im1Gray.delete();
  im2Gray.delete();
  //h.delete();
  // image_B_final_result may not have been created (warpPerspective is commented out).
  // Guard deletion to avoid ReferenceError.
  if (typeof image_B_final_result !== 'undefined' && image_B_final_result !== null) {
    try { image_B_final_result.delete(); } catch (e) { console.warn('failed to delete image_B_final_result:', e); }
  }
  mat1.delete();
  mat2.delete();
  //inlierMatches.delete();
}

// Draw matches overlay between the two displayed images.
// Good matches that are in good_inlier_matches -> green; the rest -> red.
function drawMatchesOverlay() {
  if (!good_matches_global || good_matches_global.size() === 0) return;

  strokeWeight(2);

  // We compute screen coords for lines (taking the horizontal flip into account),
  // but draw the endpoint markers inside the SAME transforms used to render the images
  // so markers align exactly with the textured images.
  for (let i = 0; i < good_matches_global.size(); ++i) {
    const m = good_matches_global.get(i);
    const px1 = points1[i * 2 + 0];
    const py1 = points1[i * 2 + 1];
    const px2 = points2[i * 2 + 0];
    const py2 = points2[i * 2 + 1];

    // determine if this match is an inlier
    let isInlier = false;
    if (good_inlier_matches && good_inlier_matches.size && good_inlier_matches.size() > 0) {
      for (let j = 0; j < good_inlier_matches.size(); ++j) {
        const im = good_inlier_matches.get(j);
        if (im.queryIdx === m.queryIdx && im.trainIdx === m.trainIdx) {
          isInlier = true;
          break;
        }
      }
    }

    let l0=applyTransform4x4(px1, py1, imageTransforms[1]);
    let l1=applyTransform4x4(px2, py2, imageTransforms[0]);
    stroke(isInlier ? 'lime' : 'red');
    circle(l0[0], l0[1], 6);
    line(l0[0], l0[1], l1[0], l1[1]);
    circle(l1[0], l1[1], 6);
  }
}

function getMatStats(Mat, name)
{
let type = Mat.type()
let channels = Mat.channels();
let cols = Mat.cols;
let rows = Mat.rows;
let depth = Mat.depth();
let baseline_colorspace = "";
let baseline_matType = "";

if (channels == 4){
baseline_colorspace = "RGBA or BGRA"
if(type == 24){baseline_matType = "CV_8UC4";}
if(type == 25){baseline_matType = "CV_8SC4";}
if(type == 26){baseline_matType = "CV_16UC4";}
if(type == 27){baseline_matType = "CV_16SC4";}
if(type == 28){baseline_matType = "CV_32SC4";}
if(type == 29){baseline_matType = "CV_32FC4";}
if(type == 30){baseline_matType = "CV_64FC4";}
}
if (channels == 3){
baseline_colorspace = "RGB, HSV or BGR";
if(type == 16){baseline_matType = "CV_8UC3";}
if(type == 17){baseline_matType = "CV_8SC3";}
if(type == 18){baseline_matType = "CV_16UC3";}
if(type == 19){baseline_matType = "CV_16SC3";}
if(type == 20){baseline_matType = "CV_32SC3";}
if(type == 21){baseline_matType = "CV_32FC3";}
if(type == 22){baseline_matType = "CV_64FC3";}
}
if (channels == 2){
baseline_colorspace = "unknown"
if(type == 8){baseline_matType = "CV_8UC2";}
if(type == 9){baseline_matType = "CV_8SC2";}
if(type == 10){baseline_matType = "CV_16UC2";}
if(type == 11){baseline_matType = "CV_16SC2";}
if(type == 12){baseline_matType = "CV_32SC2";}
if(type == 13){baseline_matType = "CV_32FC2";}
if(type == 14){baseline_matType = "CV_64FC2";}
}
if (channels == 1){
baseline_colorspace = "GRAY"
if(type == 0){baseline_matType = "CV_8UC1";}
if(type == 1){baseline_matType = "CV_8SC1";}
if(type == 2){baseline_matType = "CV_16UC1";}
if(type == 3){baseline_matType = "CV_16SC1";}
if(type == 4){baseline_matType = "CV_32SC1";}
if(type == 5){baseline_matType = "CV_32FC1";}
if(type == 6){baseline_matType = "CV_64FC1";}
}

console.log("MatName :(" + name + ") ", Mat);
console.log("   MatStats:channels=" + channels + " type:" + type + " cols:" + cols + " rows:" + rows );
console.log("   depth:" + depth + " colorspace:" + baseline_colorspace + " type:" + baseline_matType );

return;
}

// cvMatToP5Image now lives in shimage.js
// -----------------------------------------------------------------------
// imgproc - matrix math, homography validation/cleanup, alignImagePair()
// -----------------------------------------------------------------------

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
 * Converts a flat 16-element row-major 4x4 matrix into the flat 6-element
 * array [a, b, c, d, e, f] that both the HTML5 canvas API's setTransform()
 * and p5.js's applyMatrix() (2D mode) expect, where:
 *   | a c e |
 *   | b d f |
 *   | 0 0 1 |
 * Only the matrix's 2D affine part (rotation/scale/shear/translation in the
 * XY plane) survives - any Z or perspective terms are dropped.
 * @param {Array} M - flat 16-element row-major 4x4 matrix
 * @returns {Array|null} - [a, b, c, d, e, f], or null if M isn't a flat 16-element array
 */
function to2dAffine(M) {
  if (!Array.isArray(M) || M.length !== 16) return null;
  return [M[0], M[4], M[1], M[5], M[3], M[7]];
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

/**
 * Inverts a flat 6-element [a, b, c, d, e, f] 2D affine matrix - the same
 * shape alignImagePair's transform2D uses, and the one both the canvas API's
 * setTransform() and p5.js's applyMatrix() (2D mode) expect:
 *   | a c e |
 *   | b d f |
 *   | 0 0 1 |
 * @param {Array} M - flat 6-element [a, b, c, d, e, f] 2D affine matrix
 * @returns {Array|null} - flat 6-element [a, b, c, d, e, f] inverse, or null
 *   if M isn't a flat 6-element array or isn't invertible (det === 0)
 */
function invertMatrix2D(M) {
  if (!Array.isArray(M) || M.length !== 6) return null;

  const [a, b, c, d, e, f] = M;
  const det = a * d - b * c;
  if (det === 0) return null;

  const invDet = 1 / det;
  return [
    d * invDet, -b * invDet,
    -c * invDet, a * invDet,
    (c * f - d * e) * invDet, (b * e - a * f) * invDet
  ];
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
    maxScale = 3,             // max allowed scale, n - homography can be up to nx bigger or nx smaller
    maxShear = 0.3,           // max allowed shear
    maxPerspective = 0.01     // max allowed perspective distortion
  } = options;
  const minScale = 1 / maxScale;

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
 * @returns {{valid: boolean, transform: (Array|null), transform2D: (Array|null), inliers: number, reason: string}}
 */
function alignImagePair(imageA, imageB, options = {}) {
  if (!imageA || !imageB) {
    return { valid: false, transform: null, transform2D: null, inliers: 0, reason: 'imageA or imageB is null or undefined' };
  }

  try {
    Align_img(imageA, imageB);
  } catch (err) {
    return { valid: false, transform: null, transform2D: null, inliers: 0, reason: err.message };
  }

  const inliers = (good_inlier_matches && good_inlier_matches.size) ? good_inlier_matches.size() : 0;

  if (!h || h.empty() || !h.data64F) {
    return { valid: false, transform: null, transform2D: null, inliers, reason: 'No homography found' };
  }

  const check = isReasonableHomography(Array.from(h.data64F), options);
  if (!check.valid) {
    return { valid: false, transform: null, transform2D: null, inliers, reason: check.reason };
  }

  const transform = [
    h.data64F[0], h.data64F[1], 0, h.data64F[2],
    h.data64F[3], h.data64F[4], 0, h.data64F[5],
    0, 0, 1, 0,
    h.data64F[6], h.data64F[7], 0, h.data64F[8]
  ];

  // The affine part of the homography as [a, b, c, d, e, f] - the shape both
  // the canvas API's setTransform() and p5.js's applyMatrix() (2D mode)
  // expect. Perspective terms (h.data64F[6], [7]) are dropped, same as
  // to2dAffine(transform) would give - isReasonableHomography's own default
  // maxPerspective threshold already keeps those small for a valid result.
  const transform2D = [h.data64F[0], h.data64F[3], h.data64F[1], h.data64F[4], h.data64F[2], h.data64F[5]];

  return { valid: true, transform, transform2D, inliers, reason: check.reason };
}
