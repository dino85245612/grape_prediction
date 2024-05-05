/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.SystemClock;
import android.util.Base64;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import org.tensorflow.lite.examples.detection.tracking.rf_model;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

//import com.chaquo.python.PyObject;
//import com.chaquo.python.Python;
//import com.chaquo.python.android.AndroidPlatform;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 640);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    public double[] inputData = new double[11];
    OverlayView trackingOverlay;
    boolean computingDetection = false;
    int Bunch_No = 0;
    int prevBunch_No = 0;
    DecimalFormat df = new DecimalFormat("###.##");
    double predictBerryNumber2;
    double predictBerryNumber;
    int bunchAreaPrev = 0;
    int berryAreaPrev = 0;
    boolean SameBunch = true;
    boolean draw_berries = false;
    Bitmap resizeInput = null;
    int resizeWidth = 0;
    int resizeHeight = 0;
    float ScaleSize = 224.0f;//max Height or width to Scale
    Module module = null;
    Bitmap bitmap = null;
    float[] scores;
    private Integer sensorOrientation;
    private YoloV5Classifier detector;
    private long detectionProcessTime;
    private long detectionTimeBegin;
    private long detectionComputeTime;
    private long extractTimeBegin;
    private long extractProcessTime;
    private long berryPredictBegin;
    private long berryComputeTime;
    private long totalTime;
    private long startTime;
    private long totalProcessingTimeMs;
    private float framePerSecond;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private Bitmap currentBunch = null;
    private final Bitmap prevBunch = null;
    private long timestamp = 0;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private BorderedText borderedText;
    private ImageView mconvert_surface_viewMain;
    private String DetectBunch;
    private rf_model predict_3d;
    private Runnable isSameBunch;
    private Runnable plsClear;

    public DetectorActivity() {

    }

    public static Bitmap RotateBitmap(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
//        if (! Python.isStarted()) {
//            Python.start(new AndroidPlatform(DetectorActivity.this));
//        }
////        LOGGER.i("initial python pass start");

        try {
            bitmap = BitmapFactory.decodeStream(getAssets().open("Bunch_513_GT29_391.jpg"));
            int width = bitmap.getWidth();
            int height = bitmap.getHeight();
            float excessSizeRatio = width > height ? width / ScaleSize : height / ScaleSize;

            resizeWidth = (int) (width / excessSizeRatio);
            resizeHeight = (int) (height / excessSizeRatio);
            resizeInput = Bitmap.createScaledBitmap(bitmap, resizeWidth, resizeHeight, false);

            module = Module.load(assetFilePath(this, "mobile_resnet18-padding-E50.pt"));//mobile_resnet18-E140-OLD mobile_resnet18-padding-E45

        } catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }
//        startTime = SystemClock.uptimeMillis();
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizeInput, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
//        Log.i("result","result: inputTensor shape: " + inputTensor.shape().toString());
        // running the model
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

//        Log.i("result","result: scores: outputTensor: " + outputTensor);
        // getting tensor content as java array of floats
        scores = outputTensor.getDataAsFloatArray();
//        Log.i("result","result: scores: " + scores[0]);
//        totalProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
//        Log.i("result","result: scores:totalProcessingTimeMs:" + (float)(totalProcessingTimeMs));
//        framePerSecond = 1/((float)(totalProcessingTimeMs)/1000);
//        Log.i("result","result: scores:framePerSecond:" + (float)(framePerSecond));

        final int deviceIndex = deviceView.getCheckedItemPosition();
        currentDevice = deviceIndex;
        String device = "GPU"; //deviceStrings.get(deviceIndex);
        String threads = threadsTextView.getText().toString().trim();
        final int numThreads = Integer.parseInt(threads);
        currentNumThreads = numThreads;

        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        final int modelIndex = modelView.getCheckedItemPosition(); //get the check model
        final String modelString = "yolov5s-fp16-640.tflite"; //modelStrings.get(modelIndex); //get the model file name

        try { //here load the model
            detector = DetectorFactory.getDetector(getAssets(), modelString);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        if (device.equals("CPU")) {
            detector.useCPU();
        } else if (device.equals("GPU")) {
            detector.useGpu();
        } else if (device.equals("NNAPI")) {
            detector.useNNAPI();
        }
        detector.setNumThreads(numThreads);

        int cropSize = detector.getInputSize();

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
//        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

//        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

//        LOGGER.i("mUVCCameraView DA :  %dx%d", previewWidth, previewHeight);
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected int processImage() {

        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();
        final List<Classifier.Recognition> mappedRecognitionsEmpty2 =
                new LinkedList<Classifier.Recognition>(); //create empty list

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return -999;
        }
        if (Process_stage == 1) {
            tracker.trackResults(mappedRecognitionsEmpty2, currTimestamp);
            readyForNextImage();
            return -333;
        }
        computingDetection = true;

        // get the frame from camera
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground( //keep repeat loop
                new Runnable() {
                    @RequiresApi(api = Build.VERSION_CODES.N)
                    @Override
                    public void run() {
                        startTime = SystemClock.uptimeMillis();
                        ArrayList<Classifier.Recognition> selected_berry_first = new ArrayList<Classifier.Recognition>();
                        double bunch_ratio = 0;
                        double bunchBerry_ratio = 0;
                        List oneOverlap_index = new ArrayList();
                        List equalMore4Overlap_index = new ArrayList();
                        List zeroOverlap_index = new ArrayList();
                        List lessThen4Overlap_index = new ArrayList();
                        double no_ratio = 0;
                        double middle_ratio = 0;
                        double small_ratio = 0;

                        boolean bunchPosition = false;

                        detectionTimeBegin = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        detectionComputeTime = SystemClock.uptimeMillis() - detectionTimeBegin;
//                        Log.i("result","result: scores: detectionComputeTime:" + (float)(detectionComputeTime));
                        framePerSecond = 1 / ((float) (detectionComputeTime) / 1000);
//                        Log.i("result","result: scores:framePerSecond detectionComputeTime:" + (float)(framePerSecond));

                        extractTimeBegin = SystemClock.uptimeMillis();
                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(0.5f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        if (MODE == DetectorMode.TF_OD_API) {
                            minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        }

                        // 1. to separate the bbox to bunch and berry
                        //getDetectedClass : 0-bunch, 1-berry
                        ArrayList<Classifier.Recognition> filter_bunch = separateBboxClass(results, 0);
                        ArrayList<Classifier.Recognition> filter_berry = separateBboxClass(results, 1);

                        //-----For Debug----
                        ArrayList<Classifier.Recognition> list_bunch_detection = new ArrayList<Classifier.Recognition>();
                        ArrayList<Classifier.Recognition> list_berry_detection = new ArrayList<Classifier.Recognition>();
                        results.forEach((temp) -> {
                            if (temp.getDetectedClass() == 0) {
                                list_bunch_detection.add(temp);
                            } else {
                                list_berry_detection.add(temp);
                            }
                        });
//                        LOGGER.i("result from model : bunch_size --> " + list_bunch_detection.size());
//                        LOGGER.i("result from model : berry_size --> " + list_berry_detection.size());
//                        LOGGER.i("!!!!!!!!!!!!filter_bunch --> " + filter_bunch);
//                        LOGGER.i("!!!!!!!!!!!filter_berry --> " + filter_berry);
                        //------------------

                        //2. find one bunch rgbFrameBitmap
                        ArrayList<Classifier.Recognition> selected_bunch = findTargetBunch(rgbFrameBitmap, filter_bunch);

                        //check location of the bunch
//                            Params:
//                            left – The X coordinate of the left side of the rectangle
//                            top – The Y coordinate of the top of the rectangle
//                            right – The X coordinate of the right side of the rectangle
//                            bottom – The Y coordinate of the bottom of the rectangle
//                        Log.i("CHECK", "selected_bunch.get(0).getLocation(): " + selected_bunch.get(0).getLocation()
//                                + "height: " + croppedBitmap.getHeight()
//                                + "width: " + croppedBitmap.getWidth());

                        RectF bunchLocation = new RectF();
                        float rightGap = 0;
                        float bottomGap = 0;
                        Bunch_No = selected_bunch.size();
//                        LOGGER.i("processImage CA voice Bunch_No:   " + Bunch_No);

                        if (Bunch_No != 0) {
                            // checking for bunch location
                            bunchLocation = selected_bunch.get(0).getLocation();
                            int x1 = (int) bunchLocation.left;
                            int y1 = (int) bunchLocation.top;
                            int x2 = (int) bunchLocation.right;
                            int y2 = (int) bunchLocation.bottom;

                            rightGap = rgbFrameBitmap.getWidth() - x2;
                            bottomGap = rgbFrameBitmap.getHeight() - y2;

                            //Crop bunch image
                            currentBunch = Bitmap.createBitmap(rgbFrameBitmap,
                                    x1, y1, (x2 - x1), (y2 - y1));
                        }


                        // crop the detected bunch and feed into resnet18
                        if (Bunch_No != 0) {
                            float width = currentBunch.getWidth();
                            float height = currentBunch.getHeight();
                            float excessSizeRatio_width = ScaleSize / width;
                            float excessSizeRatio_height = ScaleSize / height;

                            resizeWidth = (int) (width * excessSizeRatio_width);
                            resizeHeight = (int) (height * excessSizeRatio_height);
                            currentBunch = padToSquare(currentBunch);
                            resizeInput = Bitmap.createScaledBitmap(currentBunch, resizeWidth, resizeHeight, false);

                            extractProcessTime = (long) (1 / ((float) (SystemClock.uptimeMillis() - extractTimeBegin) / 1000));

                            berryPredictBegin = SystemClock.uptimeMillis();
//                            Log.i("result","result: scores: only Bunch_No: " + Bunch_No);
                            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizeInput, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
                            // running the model
                            final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
                            // getting tensor content as java array of floats
                            scores = outputTensor.getDataAsFloatArray();
                            berryComputeTime = (long) (1 / ((float) (SystemClock.uptimeMillis() - berryPredictBegin) / 1000));

                        }


//                        // for the random forest
                        if (bunchLocation.left > 10 && bunchLocation.top > 10 && rightGap > 10 && bottomGap > 10) {
                            bunchPosition = true;
                        } else {
                            bunchPosition = false;
                            final List<Classifier.Recognition> mappedRecognitionsEmpty =
                                    new LinkedList<Classifier.Recognition>(); //create empty list
                            tracker.trackResults(mappedRecognitionsEmpty, currTimestamp);

                            trackingOverlay.postInvalidate();
                        }
//                        LOGGER.i("processImage CA voice bunchPosition:   " + bunchPosition);


                        /*
                         * for the berries bounding box
                         * */
                        if (Bunch_No != 0) {/*SameBunch == false*/
                            selected_berry_first = removeUnexpectedBerry(filter_berry, selected_bunch);

                            //9. calc each berry area
                            float largest_area = (float) 0.0;
                            List berry_area = new ArrayList();
                            List berry_area2 = new ArrayList();
                            for (int i = 0; i < selected_berry_first.size(); i++) {
                                final RectF location = selected_berry_first.get(i).getLocation();
                                float area = calc_area_bbox(location);
                                berry_area.add(area);
                            }
//                            LOGGER.i("processImage berry_area:" + berry_area);
                            berry_area2.addAll(berry_area);
                            Collections.sort(berry_area2, Collections.reverseOrder());
                            largest_area = (float) berry_area2.get(0);
                            List remove_index = new ArrayList();

//                            LOGGER.i("processImage berry_area2:" + berry_area2);
//                            LOGGER.i("processImage largest_area:" + largest_area);

                            if (berry_area2.size() > 5) {
                                for (int i = 0; i <= 5; i++) {
                                    float temp = ((float) berry_area2.get(i) - (float) berry_area2.get(5 - i)) / (float) berry_area2.get(i);
                                    if (temp > 0.5) {
                                        int index = berry_area.indexOf(berry_area2.get(i));
                                        remove_index.add(index);
                                        largest_area = (float) berry_area2.get(i + 1);
                                    }
                                }
                            }

                            double berry_size_threshold_050 = largest_area * 0.5;

                            Collections.sort(remove_index, Collections.reverseOrder());

                            //11. copy selected_berry & remove the index
                            ArrayList<Classifier.Recognition> selected_berry = new ArrayList<Classifier.Recognition>();
//                            LOGGER.i("processImage remove_index  selected_berry_first before:");
                            selected_berry = (ArrayList<Classifier.Recognition>) selected_berry_first.clone();
//                            LOGGER.i("processImage remove_index  selected_berry_first after:");


                            for (int i = 0; i < remove_index.size(); i++) {
                                int index = (int) remove_index.get(i);
//                                LOGGER.i("processImage index:" + index);
                                Classifier.Recognition remove = selected_berry.remove(index);
                            }
//                            LOGGER.i("processImage selected_berry size:" + selected_berry_first.size());
//                            LOGGER.i("processImage selected_berry size:" + selected_berry.size());

                            final List<Classifier.Recognition> mappedRecognitions =
                                    new LinkedList<Classifier.Recognition>(); //create empty list

                            for (final Classifier.Recognition result : selected_bunch) {
                                final RectF location = result.getLocation(); //get the bounding box
                                if (location != null) { //(location != null && result.getConfidence() >= minimumConfidence) //check for the confidence score
                                    if (result.getDetectedClass() == 0) { //for bunch
                                        canvas.drawRect(location, paint);
                                    }
                                    cropToFrameTransform.mapRect(location);
                                    result.setLocation(location);
                                    mappedRecognitions.add(result); //those result that pass the confidence threshold
                                }
                            }

                            if (draw_berries) {
                                for (final Classifier.Recognition result : selected_berry) {
                                    final RectF location = result.getLocation(); //get the bounding box
                                    if (location != null) { //(location != null && result.getConfidence() >= minimumConfidence) //check for the confidence score
                                        if (result.getDetectedClass() == 0) { //for bunch
                                            canvas.drawRect(location, paint);
                                        }
                                        cropToFrameTransform.mapRect(location);
                                        result.setLocation(location);
                                        mappedRecognitions.add(result); //those result that pass the confidence threshold
                                    }
                                }
                            }

                            tracker.trackResults(mappedRecognitions, currTimestamp);
                            trackingOverlay.postInvalidate();

                        }


                        if (Bunch_No != 0 && bunchPosition) {//&& SameBunch == false
                            SharedPreferences sharedPref = getSharedPreferences("Grape", Context.MODE_PRIVATE);
                            Boolean moreThan;
                            int differenceResult = (int) scores[0] - selected_berry_first.size();
                            moreThan = differenceResult >= 0;
                            Log.i("result", "result: (int)predictBerryNumber: " + (int) predictBerryNumber);
                            Log.i("result", "result: (int)scores[0]: " + (int) scores[0]);
                            Log.i("result", "result: selected_berry_first.size(): " + selected_berry_first.size());
                            Log.i("result", "result: differenceResult: " + differenceResult);
                            Log.i("result", "result: moreThan: " + moreThan);

                            String folderName = sharedPref.getString("inputFolderName", "noInput");
                            if (moreThan) {
//                                try {
//                                    String directoryPath = Environment.getExternalStorageDirectory().toString() + "/grape_exp/" + folderName;
////                                Log.i("result", "result: directoryPath: " + directoryPath);
//                                    File dir = new File(directoryPath);
//                                    if(!dir.exists()){
//                                        dir.mkdirs();
//                                    }
//
//                                    String timeStamp = new SimpleDateFormat("MMdd_HHmmss").format(new Date());
//                                    String file_name = "IMG_" + timeStamp + "_D-" + selected_berry_first.size() + "_P-" + (int)scores[0] + ".png";
//                                    File file = new File(dir, file_name);
//                                    FileOutputStream fOutputStream = new FileOutputStream(file);
//
////                                Log.i("result", "result: file_name: " + file_name);
//                                    croppedBitmap.compress(Bitmap.CompressFormat.PNG, 100, fOutputStream);
//
//                                    fOutputStream.flush();
//                                    fOutputStream.close();
//
//                                    MediaStore.Images.Media.insertImage(getContentResolver(), file.getAbsolutePath(), file.getName(), file.getName());
//                                } catch (Exception e) {
////                                Log.i("result", "result: directoryPath: failed" );
//                                    e.printStackTrace();
//                                }


                                if (filter_berry.size() > 65) {
                                    predictBerryNumber = filter_berry.size();
                                }
//                                else if (filter_berry.size() < 20){
//                                    predictBerryNumber = filter_berry.size();}
                                else {
                                    totalProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                                    predictBerryNumber = (int) scores[0];
                                    framePerSecond = 1 / ((float) (totalProcessingTimeMs) / 1000);
//                                Log.i("result", "result: resnet18 prediction: " + predictBerryNumber);
//                                Log.i("result","result: extractProcessTime: " + (float)extractProcessTime + "fps");
//                                Log.i("result", "result: resnet18 prediction time: " + (float)berryComputeTime + "fps");
//                                Log.i("result","result: overall computational time: " + framePerSecond + "fps");
                                }
                            } else {
                                if (filter_berry.size() > 55) {
                                    predictBerryNumber = filter_berry.size();
                                }
                            }


                        } else if (Bunch_No != 0) { // detected bunch but out of location
                            predictBerryNumber = -777;
                            totalProcessingTimeMs = 0;
                            framePerSecond = 0;
                        } else { // detect nothing from the frame
                            predictBerryNumber = 0;
                            totalProcessingTimeMs = 0;
                            framePerSecond = 0;
                            SameBunch = false;
                        }

                        computingDetection = false;
                        if (Bunch_No != 0) {
                            prevBunch_No = 1;
                        } else {
                            prevBunch_No = 0;
                        }

                    }


                });
//        LOGGER.i("processImage get_result-----(int)predictBerryNumber----: " + (int)predictBerryNumber);
        return (int) predictBerryNumber;
    }

    public Bitmap padToSquare(Bitmap bmp) {
        int width = bmp.getWidth();
        int height = bmp.getHeight();
        int max_wh = Math.max(width, height);
        int diff_w = max_wh - width;
        int diff_h = max_wh - height;
        int padding_w = diff_w / 2;  // Padding on the sides
        int padding_h = diff_h / 2;  // Padding on the top and bottom

        // Create a new bitmap with the desired size
        Bitmap bmpWithBorder = Bitmap.createBitmap(max_wh, max_wh, bmp.getConfig());
//        bmpWithBorder.eraseColor(Color.green(139));  // Fill the bitmap with the color of the padding
        bmpWithBorder.eraseColor(Color.RED);

        // Create a canvas to draw on the new bitmap
        Canvas canvas = new Canvas(bmpWithBorder);

        // Draw the original bitmap onto the new bitmap with the calculated padding
        canvas.drawBitmap(bmp, padding_w, padding_h, null);

        return bmpWithBorder;
    }

    private String getStringImage(Bitmap bitmap) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);
        byte[] imageBytes = baos.toByteArray();
        String encodedImage = android.util.Base64.encodeToString(imageBytes, Base64.DEFAULT);
        return encodedImage;
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public ArrayList<Classifier.Recognition> separateBboxClass(List<Classifier.Recognition> results,
                                                               int classID) {
        ArrayList<Classifier.Recognition> bunch_boxes = new ArrayList<Classifier.Recognition>();

        List<Classifier.Recognition> prediction = results;
        //LOGGER.i("prediction:" + prediction);
        //1. prediction already filter out bbox conf < 0.3 (so here no need)
        //2. separate bunch and berry bbox: getDetectedClass : 0-bunch, 1-berry

        //--------New Code------
        final float[] x0 = {0};
        final float[] x1 = {0};
        final float[] y0 = {0};
        final float[] y1 = {0};
//        LOGGER.i("prediction size:" + prediction.size());
        prediction.forEach((temp) -> {
//            LOGGER.i("prediction temp:" + temp);
            if (temp.getTitle().equals("Bunch")) {
                x0[0] = temp.getLocation().left;
                x1[0] = temp.getLocation().right;
                y0[0] = temp.getLocation().top;
                y1[0] = temp.getLocation().bottom;
//                LOGGER.i("prediction positionBunch: x0 -> " + x0[0] + " x1 -> " + x1[0] + " y0 -> " + y0[0] + " y1 -> " + y1[0]);
            }
            if (classID == 1) {
                //If berry is out of bunch not put into the list
                if (temp.getLocation().left >= x0[0] && temp.getLocation().right < x1[0]
                        && temp.getLocation().top >= y0[0] && temp.getLocation().bottom < y1[0]
                        && temp.getDetectedClass() == classID) {
                    bunch_boxes.add(temp);
                }
            } else if (classID == 0 && temp.getDetectedClass() == classID) {
                //Try to check which bunch should focus.
                // Get the screen dimensions
//                DisplayMetrics displayMetrics = new DisplayMetrics();
//                getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
//                int screenWidth = displayMetrics.widthPixels;
//                int screenHeight = displayMetrics.heightPixels;

                // Calculate the center of the screen
//                float screenCenterX = screenWidth / 2;
//                float screenCenterY = screenHeight / 2;

                bunch_boxes.add(temp);
            }
            //-------------------

            //--------Old code--------
//            if (temp.getDetectedClass() == classID) {
//                bunch_boxes.add(temp);
//            }
        });
        LOGGER.i("bunch_boxes.size():" + bunch_boxes.size());
        return bunch_boxes;
    }

    public ArrayList<Classifier.Recognition> findTargetBunch(Bitmap cropBitmap,
                                                             ArrayList<Classifier.Recognition> arrayList) {
        ArrayList<Classifier.Recognition> bunch_bboxs = new ArrayList<Classifier.Recognition>();

        if (arrayList.size() == 1) {
            return arrayList;
        } else {
            // 1. filter bunch overlap
            //bunch_bboxs = (ArrayList<Classifier.Recognition>) filter_overlap_bunch(arrayList);
            // 2. find the middle bunch
            // xxx = find_middle_bunch(cropBitmap, bunch_bboxs);
        }

        return arrayList;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public ArrayList<Classifier.Recognition> removeUnexpectedBerry(ArrayList<Classifier.Recognition> filter_berry, ArrayList<Classifier.Recognition> selected_bunch) {
        final float[] iou = new float[1];
        //copy filter_berry
        ArrayList<Classifier.Recognition> selected_berry = new ArrayList<Classifier.Recognition>();

        //1. perform ious with selected bunch coor (iou !=0)
        //2. remove berry area that is too big (iou < 0.1)
        //LOGGER.i("remove_unexpected_berry  berry_bbox:" + filter_berry);
        //LOGGER.i("remove_unexpected_berry  selected_bunch:" + selected_bunch.getClass());

        selected_bunch.forEach((temp) -> {
            //LOGGER.i("temp  :" + temp.getLocation());
            filter_berry.forEach((temp2) -> {
                //LOGGER.i("temp2 :" + temp2.getLocation());

                iou[0] = box_iou(temp2.getLocation(), temp.getLocation());
                //LOGGER.i("iou[0]:" + iou[0]);
                if ((iou[0] != 0.0) && (iou[0] < 0.1)) {
                    selected_berry.add(temp2);
                }

            });


        });
//        Log.i("CHECK", "method selected_berry.size(): " + selected_berry.size());

        return selected_berry;
    }

    public float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    public float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    public float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    public float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    public Object toBlackBitmap(Bitmap cropBitmap) {
        Bitmap blackBitmap;
        int width = cropBitmap.getWidth();
        int height = cropBitmap.getHeight();
        int[] pixels = new int[width * height];
        Bitmap.Config conf = Bitmap.Config.RGB_565; // see other conf types
        blackBitmap = Bitmap.createBitmap(width, height, conf); // this creates a MUTABLE bitmap
        Canvas canvas = new Canvas(blackBitmap);
        canvas.drawColor(Color.BLACK);
        //content.draw(canvas);
        return blackBitmap;
    }

    public Bitmap drawWhiteRect(Bitmap blackBitmap, ArrayList<Classifier.Recognition> bboxes) {
        // copy the bitmap
        Bitmap copyBitmap = blackBitmap.copy(blackBitmap.getConfig(), true);
        final Canvas canvas = new Canvas(copyBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.WHITE);
        paint.setStyle(Paint.Style.FILL);

        for (final Classifier.Recognition result : bboxes) {
            final RectF location = result.getLocation();
            if (location != null) {
                canvas.drawRect(location, paint);
            }
        }

        return copyBitmap;
    }

    public Bitmap drawWhiteCircle(Bitmap blackBitmap, ArrayList<Classifier.Recognition> bboxes) {
        // copy the bitmap
        Bitmap copyBitmap = blackBitmap.copy(blackBitmap.getConfig(), true);
        final Canvas canvas = new Canvas(copyBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.WHITE);
        paint.setStyle(Paint.Style.FILL);

        for (final Classifier.Recognition result : bboxes) {
            final RectF location = result.getLocation();
            if (location != null) {
                float x_center = (location.left + location.right) / 2;   // x center
                float y_center = (location.top + location.bottom) / 2;   // y center
                float square_width = location.right - location.left;      // width
                float square_height = location.bottom - location.top;     // height
                float radius;
                if (square_width >= square_height) {
                    radius = square_width / 2;
                } else {
                    radius = square_height / 2;
                }

//                int square_center = (int(x_center), int(y_center))
                canvas.drawCircle(x_center, y_center, radius, paint);
            }
        }

        return copyBitmap;
    }


    public float calc_ratio(int a, int b) {
        return (float) a / b;
    }

    public float calc_area_bbox(RectF a) {
        //float area = (a.right - a.left) * (a.bottom - a.top);
        return (a.right - a.left) * (a.bottom - a.top);
    }

    public float findLargest(float a, float b) {
        if (a > b) {
            return a;
        } else return b;
    }

    public boolean getBunchIdea() {
        isSameBunch.run();
        return SameBunch;
    }

    public void clearBbox() {
        plsClear.run();
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    public enum DetectorMode {
        TF_OD_API
    }

}
