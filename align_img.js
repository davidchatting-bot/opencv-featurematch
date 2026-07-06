var canvas;
var inputImageA = null, inputImageB = null;
var points1 = [];
var points2 = [];
var good_inlier_matches;
var h;
var good_matches_global = null; // store matches so draw() can render them

function Align_img(image_element_a, image_element_b) {
    if(!image_element_a || !image_element_b) return null;

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
  inputImageB = createImage(resultSize.width, resultSize.height);
  cvMatToP5Image(im2, inputImageB);
  
  let im1 = cv.imread(image_element_b);
  
  resultSize = im1.size();
  inputImageA = createImage(resultSize.width, resultSize.height);
  cvMatToP5Image(im1, inputImageA);

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