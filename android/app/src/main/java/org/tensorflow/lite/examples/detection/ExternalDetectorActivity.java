package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

//import com.chaquo.python.PyObject;
//import com.chaquo.python.Python;
import com.serenegiant.usb.CameraDialog;
import com.serenegiant.usb.IFrameCallback;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import org.tensorflow.lite.examples.detection.tracking.rf_model;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class ExternalDetectorActivity extends CameraDemo implements CameraDialog.CameraDialogParent, IFrameCallback {
    private static final Logger LOGGER = new Logger();

    private static final DetectorActivity.DetectorMode MODE = DetectorActivity.DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 640);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private YoloV5Classifier detector;

    private long lastProcessingTimeMs;
    private long startTime;
    private long totalProcessingTimeMs;
    private long totalTime;
    private float framePerSecond;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    private String DetectBunch;
    int Bunch_No;
    //Detected_Berry	bunch_ratio	bunchBerry_ratio	partially_overlap	no overlap %
    public double inputData[] = new double[5];
    DecimalFormat df = new DecimalFormat("###.##");
    private rf_model predict_3d;
    double predictBerryNumber;
    int previewWidth;
    int previewHeight;

    private ImageView uvc_image2;
    @Override
    public void onYOLOChosen() {
        Log.i("CameraDemo - ExternalDA", "onYOLOChosen");
        String device = "GPU"; //deviceStrings.get(deviceIndex);
        String threads = "1";
        final int numThreads = Integer.parseInt(threads);
        currentNumThreads = numThreads;

        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);
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
//        LOGGER.i("mUVCCameraView EDA cropSize %dx%d", cropSize, cropSize);

        previewWidth = getPreviewWidth();//getPreviewWidth();
        previewHeight = getPreviewHeight();//getPreviewHeight();
//        LOGGER.i("mUVCCameraView EDA previewWidth: %d", previewWidth);
//        LOGGER.i("mUVCCameraView EDA previewHeight: %d", previewHeight);

        sensorOrientation = 0 - getScreenOrientation();
//        LOGGER.i("mUVCCameraView EDA getScreenOrientation() : %d", getScreenOrientation());
//        LOGGER.i("mUVCCameraView EDA orientation : %d", sensorOrientation);

        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

//        LOGGER.i("mUVCCameraView EDA frameToCropTransform : %d", frameToCropTransform);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
//                        LOGGER.i("mUVCCameraView EDA canvas.getHeight(): " + canvas.getHeight());
//                        LOGGER.i("mUVCCameraView EDA canvas.getWidth(): " + canvas.getWidth());

                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

//        Log.i("CameraDemo - ExternalDA", "onYOLOChosen");
        // according to uvc camera: width 640, height 480, orientation 0
        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
//        uvc_image2 = (ImageView)findViewById(R.id.convert_surface_view2);
    }


    protected int processImage() {
//        LOGGER.i("mUVCCameraView EDA processImage");
//        uvc_image2 = (ImageView)findViewById(R.id.convert_surface_view2);
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return 0;
        }
        computingDetection = true;
//        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap = getRgbBytes();
//        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
//        rgbFrameBitmap = tempBitmap;
//        uvc_image2.setImageBitmap(rgbFrameBitmap);
//        LOGGER.i("mUVCCameraView EDA rgbFrameBitmap at size %dx%d", rgbFrameBitmap.getWidth(), rgbFrameBitmap.getHeight());

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }


//        uvc_image2.setImageBitmap(croppedBitmap);

        runInBackground( //keep repeat loop
                new Runnable() {
                    @RequiresApi(api = Build.VERSION_CODES.N)
                    @Override
                    public void run() {
                        ArrayList<Classifier.Recognition> selected_berry = new ArrayList<Classifier.Recognition>();
                        double bunch_ratio = 0;
                        double bunchBerry_ratio = 0;
                        List partially_overlap_index = new ArrayList();
                        List no_overlap_index = new ArrayList();
                        float no_ratio = (float) 0.0;
                        double inputData[] = new double[5];
                        DecimalFormat df = new DecimalFormat("###.##");
                        boolean bunchPosition = false;

//                        LOGGER.i("Running detection on image " + currTimestamp);
                        startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

//                        Log.i("CHECK", "results.size(): " + results.size());

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(0.f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }


                        // 1. to separate the bbox to bunch and berry
                        //getDetectedClass : 0-bunch, 1-berry
                        ArrayList<Classifier.Recognition> filter_bunch = separateBboxClass(results, 0);
                        ArrayList<Classifier.Recognition> filter_berry = separateBboxClass(results, 1);
//                        Log.i("CHECK", "results: " + results);
//                        Log.i("CHECK", "prediction bunch_bboxes: " + filter_bunch);
//                        //Log.i("CHECK", "prediction bunch_bboxes: " + filter_bunch.size());
//                        Log.i("CHECK", "prediction filter_berry: " + filter_berry);
//                        Log.i("CHECK", "prediction filter_berry: " + filter_berry.size());

                        //2. find one bunch
                        ArrayList<Classifier.Recognition> selected_bunch = findTargetBunch(croppedBitmap, filter_bunch);
                        //LOGGER.i("selected_bunch:" + selected_bunch);


                        //check location of the bunch
//                            Params:
//                            left – The X coordinate of the left side of the rectangle
//                            top – The Y coordinate of the top of the rectangle
//                            right – The X coordinate of the right side of the rectangle
//                            bottom – The Y coordinate of the bottom of the rectangle
//                        Log.i("CHECK", "selected_bunch.get(0).getLocation(): " + selected_bunch.get(0).getLocation()
//                                + "height: " + croppedBitmap.getHeight()
//                                + "width: " + croppedBitmap.getWidth());

                        Bunch_No = selected_bunch.size();

                        if (Bunch_No != 0){
                            final RectF bunchLocation = selected_bunch.get(0).getLocation();
                            float rightGap = croppedBitmap.getWidth() - bunchLocation.right;
                            float bottomGap = croppedBitmap.getHeight() - bunchLocation.bottom;
                        }

//                        if (bunchLocation.left > 20 && bunchLocation.top > 20 && rightGap > 20 && bottomGap > 20){
//                            bunchPosition = true;
//                        }else{
//                            bunchPosition = false;
//                            final List<Classifier.Recognition> mappedRecognitionsEmpty =
//                                new LinkedList<Classifier.Recognition>(); //create empty list
//                            tracker.trackResults(mappedRecognitionsEmpty, currTimestamp);
//                            trackingOverlay.postInvalidate();
//
//                        };
//                        Log.i("CHECK", "rightGap: " + rightGap + "bottomGap: " + bottomGap);
//                        Log.i("CHECK", "bunchPosition: " + bunchPosition);



                        //3. check is there recognise a bunch
//                        if (Bunch_No != 0 && bunchPosition == true) {

                        //4. remove berry out of the selected bunch
                        //ArrayList<Classifier.Recognition> selected_berry;
                        selected_berry = removeUnexpectedBerry(filter_berry, selected_bunch);
//                        Log.i("CHECK", "selected_berry.size(): " + selected_berry.size());
//                        Log.i("CHECK", "selected_berry: " + selected_berry);


                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>(); //create empty list

//                        for (final Classifier.Recognition result : selected_bunch) {
//                            final RectF location = result.getLocation(); //get the bounding box
//                            if (location != null) { //(location != null && result.getConfidence() >= minimumConfidence) //check for the confidence score
//                                if (result.getDetectedClass() == 0) { //for bunch
//                                    canvas.drawRect(location, paint);
//                                }
//
//                                cropToFrameTransform.mapRect(location);
//
//                                result.setLocation(location);
//                                mappedRecognitions.add(result); //those result that pass the confidence threshold
//                            }
//                        }
//                        for (final Classifier.Recognition result : selected_berry) {
//                            final RectF location = result.getLocation(); //get the bounding box
//                            if (location != null) { //(location != null && result.getConfidence() >= minimumConfidence) //check for the confidence score
//                                if (result.getDetectedClass() == 0) { //for bunch
//                                    canvas.drawRect(location, paint);
//                                }
//
//                                cropToFrameTransform.mapRect(location);
//
//                                result.setLocation(location);
//                                mappedRecognitions.add(result); //those result that pass the confidence threshold
//                            }
//                        }
//
////                        Log.i("CHECK", "mappedRecognitions.size(): " + mappedRecognitions.size());
//
//                        tracker.trackResults(mappedRecognitions, currTimestamp);
//                        trackingOverlay.postInvalidate();


                        //5. images area
                        int image_area = croppedBitmap.getWidth() * croppedBitmap.getHeight();
                        //int image_area = sourceBitmap.getWidth() * sourceBitmap.getHeight();

                        OpenCVLoader.initDebug();
                        //6. mask selected bunch into black bitmap, and get the area of white bbox
                        //a. create an black bitmap first, from cropBitmap
                        Bitmap blackBitmap = (Bitmap) toBlackBitmap(croppedBitmap);
                        //imageView2.setImageBitmap(blackBitmap);
                        //b. mask the bunch bbox into the image
                        Bitmap bunch_mask = drawWhiteRect(blackBitmap, selected_bunch);
                        //imageView2.setImageBitmap(bunch_mask);
                        //LOGGER.i("bunch_mask" + bunch_mask.getClass());
                        //c. calculate the area of bunch
                        Mat mat_bunch_mask = new Mat();
                        org.opencv.android.Utils.bitmapToMat(bunch_mask, mat_bunch_mask);
                        Imgproc.cvtColor(mat_bunch_mask, mat_bunch_mask, Imgproc.COLOR_BGR2GRAY);
                        int bunch_areas = Core.countNonZero(mat_bunch_mask);
                        //LOGGER.i("bunch_areas:" + bunch_areas);

                        //7. bunch_ratio
                        bunch_ratio = (calc_ratio(bunch_areas, image_area)) * 100;

                        //8. berry areas
                        //b. mask the berry bbox into the image
                        Bitmap berry_mask = drawWhiteRect(blackBitmap, selected_berry);
                        //imageView2.setImageBitmap(berry_mask);
                        //LOGGER.i("berry_mask" + berry_mask.getClass());
                        //c. calculate the area of bunch
                        Mat mat_berry_mask = new Mat();
                        org.opencv.android.Utils.bitmapToMat(berry_mask, mat_berry_mask);
                        Imgproc.cvtColor(mat_berry_mask, mat_berry_mask, Imgproc.COLOR_BGR2GRAY);
                        int berry_areas = Core.countNonZero(mat_berry_mask);
                        //LOGGER.i("berry_areas:" + berry_areas);

                        //8a bunchberry ratio
                        bunchBerry_ratio = (calc_ratio(berry_areas, bunch_areas)) * 100;


                        //9. calc each berry area
                        float largest_area = (float) 0.0;
                        List berry_area = new ArrayList();
                        for (int i = 0; i < selected_berry.size(); i++) {
                            final RectF location = selected_berry.get(i).getLocation();
                            float area = calc_area_bbox(location);
                            berry_area.add(area);
                            if (i == 0) {
                                largest_area = area;
                            } else {
                                largest_area = findLargest(largest_area, area);
                            }
                        }
                        //LOGGER.i("berry_area:" + berry_area);
                        //LOGGER.i("largest_area:" + largest_area);

                        //10. remove smallest area berry
                        float ratio_area = (float) (largest_area * 0.4);
                        List remove_index = new ArrayList();
                        for (int i = 0; i < berry_area.size(); i++) {
                            //LOGGER.i("berry_area:" + i + "i"+ berry_area.get(i));
                            float temp = (float) berry_area.get(i);
                            if (temp >= ratio_area) continue;
                            else {
                                remove_index.add(i);
                            }
                        }
                        Collections.sort(remove_index, Collections.reverseOrder());
                        //LOGGER.i("remove_index:" + remove_index.size());
                        //LOGGER.i("remove_index:" + remove_index);

                        //11. copy selected_berry & remove the index
                        ArrayList<Classifier.Recognition> remain_berry = new ArrayList<Classifier.Recognition>();
                        remain_berry = (ArrayList<Classifier.Recognition>) selected_berry.clone();
                        for (int i = 0; i < remove_index.size(); i++) {
                            int index = (int) remove_index.get(i);
                            //LOGGER.i("remove index:" + index);
                            Classifier.Recognition remove = remain_berry.remove(index);
                        }
                        //LOGGER.i("selected_berry size:" + selected_berry.size());
                        //LOGGER.i("remain berry size:" + remain_berry.size());

                        //12. partially_overlap,	no overlap %

                        final float[] iou = new float[1];
                        int overlap_no = 0;
                        for (int i = 0; i < remain_berry.size(); i++) {
                            final RectF location = remain_berry.get(i).getLocation();
                            for (int j = 0; j < remain_berry.size(); j++) {
                                if (i != j) {
                                    final RectF location2 = remain_berry.get(j).getLocation();
                                    iou[0] = box_iou(location, location2);
                                    //LOGGER.i("iou[0]:" + iou[0]);
                                    if (iou[0] != 0.0) {
                                        overlap_no += 1;
                                    }

                                }

                            }
                            if (overlap_no == 0) {
                                no_overlap_index.add(i);
                            } else if (overlap_no >= 4) {
                                partially_overlap_index.add(i);
                            }
                        }
                        //LOGGER.i("no_overlap_index:" + no_overlap_index);
                        //LOGGER.i("no_overlap_index size:" + no_overlap_index.size());
                        //LOGGER.i("partially_overlap_index:" + partially_overlap_index);
                        //LOGGER.i("partially_overlap_index size:" + partially_overlap_index.size());


                        if (no_overlap_index.size() != 0) {
                            no_ratio = (float) (no_overlap_index.size() / selected_berry.size()) * 100;
                        } else no_ratio = (float) 0.0;

                            /*
                            LOGGER.i("no_ratio:" + no_ratio);
                            LOGGER.i("image_area:" + image_area);
                            LOGGER.i("bunch_areas:" + bunch_areas);
                            LOGGER.i("bunch_ratio:" + bunch_ratio);
                            LOGGER.i("bunchBerry_ratio:" + bunchBerry_ratio);

                             */

                        inputData[0] = selected_berry.size();//36;
                        inputData[1] = Double.parseDouble(df.format(bunch_ratio));//(double) 10.48;
                        inputData[2] = Double.parseDouble(df.format(bunchBerry_ratio));//(double) 49.64;
                        inputData[3] = partially_overlap_index.size();//(double) 3.0;
                        inputData[4] = no_ratio;//(double) 11.11;
//                        }
//                        else{
//                            inputData[0] = 0;
//                            inputData[1] = (double) 0.0;
//                            inputData[2] = (double) 0.0;
//                            inputData[3] = (double) 0.0;
//                            inputData[4] = (double) 0.0;
//                        };



                        if(Bunch_No != 0){
//                            predictBerryNumber = predict_3d.score(inputData);
//                            LOGGER.i("DetectorActivity java prdict:" + predictBerryNumber2);
                            totalProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
//                            LOGGER.i("Yolo execution time:" + (float)(lastProcessingTimeMs));
//                            LOGGER.i("totalProcessingTimeMs:" + (float)(totalProcessingTimeMs));
                            framePerSecond = 1/((float)(totalProcessingTimeMs)/1000);
//                            LOGGER.i("framePerSecond: before" + (int)framePerSecond);

//                            Python py = Python.getInstance();
//                            PyObject pyObject = py.getModule("plot").callAttr("plot", inputData[0],inputData[1],
//                                    inputData[2],inputData[3],inputData[4]);

                            // 将Python返回值换为Java中的Integer类型
//                            predictBerryNumber = pyObject.toJava(double.class);
//                            LOGGER.i("DetectorActivity chaquopy result" + pyObject.toString());
                            totalProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
//                            LOGGER.i("Yolo execution time:" + (float)(lastProcessingTimeMs));
//                            LOGGER.i("totalProcessingTimeMs:" + (float)(totalProcessingTimeMs));
                            framePerSecond = 1/((float)(totalProcessingTimeMs)/1000);
//                            LOGGER.i("framePerSecond: after" + (int)framePerSecond);
                            if (filter_berry.size() > 65){
                                predictBerryNumber = filter_berry.size();
                            }
                        }else{
                            predictBerryNumber = 0.0;
                            totalProcessingTimeMs = 0;
                            framePerSecond = 0;
                        }

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
//                                        showFrameInfo(totalProcessingTimeMs + "ms");
//                                        showCropInfo(String.valueOf((int)predictBerryNumber));
//                                        showInference((int)framePerSecond + "fps");
                                    }
                                });
                    }


                });

        return (int)predictBerryNumber;
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking_external;
    }

    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

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
        //LOGGER.i("prediction:" + prediction);
        //LOGGER.i("prediction size:" + prediction.size());
        prediction.forEach((temp) ->{
            //LOGGER.i("prediction temp:" + temp);
            if (temp.getDetectedClass() == classID) {
                bunch_boxes.add(temp);
            }
        });

        return bunch_boxes;
    }

    public ArrayList<Classifier.Recognition> findTargetBunch(Bitmap cropBitmap,
                                                             ArrayList<Classifier.Recognition> arrayList) {
        ArrayList<Classifier.Recognition> bunch_bboxs = new ArrayList<Classifier.Recognition>();

        if (arrayList.size() == 1){
            return arrayList;
        }
        else{
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

        selected_bunch.forEach((temp)->{
            //LOGGER.i("temp  :" + temp.getLocation());
            filter_berry.forEach((temp2) ->{
                //LOGGER.i("temp2 :" + temp2.getLocation());

                iou[0] = box_iou(temp2.getLocation(),temp.getLocation());
                //LOGGER.i("iou[0]:" + iou[0]);
                if ((iou[0] != 0.0) && (iou[0] < 0.1)){
                    selected_berry.add(temp2);
                }

            });


        });
        Log.i("CHECK", "method selected_berry.size(): " + selected_berry.size());

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

    public float calc_ratio(int a, int b) {
        return (float)a/b;
    }

    public float calc_area_bbox(RectF a) {
        //float area = (a.right - a.left) * (a.bottom - a.top);
        return (a.right - a.left) * (a.bottom - a.top);
    }

    public float findLargest(float a, float b) {
        if (a > b){ return a; }
        else return b;
    }

    //    public void onPreviewSizeChosen(final Size size, final int rotation) {
//        String device = "GPU"; //deviceStrings.get(deviceIndex);
//        String threads = "1";
//        final int numThreads = Integer.parseInt(threads);
//        currentNumThreads = numThreads;
//
//        final float textSizePx =
//                TypedValue.applyDimension(
//                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
//        borderedText = new BorderedText(textSizePx);
//        borderedText.setTypeface(Typeface.MONOSPACE);
//
//        tracker = new MultiBoxTracker(this);
//        final String modelString = "yolov5s-fp16-640.tflite"; //modelStrings.get(modelIndex); //get the model file name
//        try { //here load the model
//            detector = DetectorFactory.getDetector(getAssets(), modelString);
//        } catch (final IOException e) {
//            e.printStackTrace();
//            LOGGER.e(e, "Exception initializing classifier!");
//            Toast toast =
//                    Toast.makeText(
//                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
//            toast.show();
//            finish();
//        }
//
//        if (device.equals("CPU")) {
//            detector.useCPU();
//        } else if (device.equals("GPU")) {
//            detector.useGpu();
//        } else if (device.equals("NNAPI")) {
//            detector.useNNAPI();
//        }
//        detector.setNumThreads(numThreads);
//
//        int cropSize = detector.getInputSize();
//
//        previewWidth = size.getWidth();
//        previewHeight = size.getHeight();
//
//        sensorOrientation = rotation - getScreenOrientation();
//        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);
//
//        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
//        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
//        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);
//
//        frameToCropTransform =
//                ImageUtils.getTransformationMatrix(
//                        previewWidth, previewHeight,
//                        cropSize, cropSize,
//                        sensorOrientation, MAINTAIN_ASPECT);
//
//        cropToFrameTransform = new Matrix();
//        frameToCropTransform.invert(cropToFrameTransform);
//
//        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
//        trackingOverlay.addCallback(
//                new OverlayView.DrawCallback() {
//                    @Override
//                    public void drawCallback(final Canvas canvas) {
//                        tracker.draw(canvas);
//                        if (isDebug()) {
//                            tracker.drawDebug(canvas);
//                        }
//                    }
//                });
//
////        Log.i("CameraDemo - ExternalDA", "onPreviewSizeChosen");
//        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
//    }

}
