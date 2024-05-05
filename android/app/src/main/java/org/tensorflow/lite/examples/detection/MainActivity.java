package org.tensorflow.lite.examples.detection;

import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ConfigurationInfo;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

//import com.chaquo.python.PyObject;
//import com.chaquo.python.Python;
//import com.chaquo.python.android.AndroidPlatform;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import org.tensorflow.lite.examples.detection.tracking.PreprocessFeature;
import org.tensorflow.lite.examples.detection.tracking.rf_model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    //with final keyword, define value is unchangeable
    public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private long lastProcessingTimeMs;
    private TextView log_information;
    private static final Logger LOGGER = new Logger();

    private Classifier detector;
    public static final int TF_OD_API_INPUT_SIZE = 640;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "yolov5s-fp16-640.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt";

    private static String DetectBunch = "No"; //Default
    private  static int Bunch_No = 0; //Default

    // Minimum detection confidence to track a detection.
    private static final boolean MAINTAIN_ASPECT = true;
    private Integer sensorOrientation = 90;


    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private OverlayView trackingOverlay;

    protected int previewWidth = 0;
    protected int previewHeight = 0;

    private Bitmap sourceBitmap;
    private Bitmap cropBitmap;

    private Button cameraButton, detectButton, btnExternalCamera;
    private ImageView imageView;
    //private ImageView imageView2;
    private rf_model predict_3d;
    private PreprocessFeature RunCollect;
    private long t1Time, inferTime, preProcess_time;
    double result;
    private List<Classifier.Recognition> copyResults;
    protected TextView cropValueTextView;

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraButton = findViewById(R.id.cameraButton);
        detectButton = findViewById(R.id.detectButton);
        btnExternalCamera = findViewById(R.id.btnExternalCamera);
        imageView = findViewById(R.id.imageView);
        log_information = findViewById(R.id.log_infor);
        cropValueTextView = findViewById(R.id.crop_info);

        //inputData = new double[5];// create int array to store int pixel
        double inputData[] = new double[5];

        inputData[0] = 0;
        inputData[1] = (double) 0.0;
        inputData[2] = (double) 0.0;
        inputData[3] = (double) 0.0;
        inputData[4] = (double) 0.0;

//        inputData[0] = 24;
//        inputData[1] = (double) 869658.0;
//        inputData[2] = (double) 420638.0;
//        inputData[3] = (double) 41.94;
//        inputData[4] = (double) 48.37;

//        inputData[0] = 26;
//        inputData[1] = (double) 423905;
//        inputData[2] = (double) 176155;
//        inputData[3] = (double) 20.44;
//        inputData[4] = (double) 41.56;

//        inputData[0] = 37;
//        inputData[1] = (double) 75151;
//        inputData[2] = (double) 25301;
//        inputData[3] = (double) 11.29;
//        inputData[4] = (double) 33.67;

        //run once
        /*
        LOGGER.i("Initialize RF model run once with 0 data:");
        t1Time = System.nanoTime ();//System.currentTimeMillis();uptimeMilliselapsedRealtime
        result = predict_3d.score(inputData);
        inferTime = System.nanoTime  () - t1Time;
        LOGGER.i("inferTime:" + inferTime);
        LOGGER.i("Outputs:" + result);
        */
        // "context" must be an Activity, Service or Application object from your app.
//        LOGGER.i("initial python start");
//        if (! Python.isStarted()) {
//            Python.start(new AndroidPlatform(this));
////            LOGGER.i("initial python enter start");
//        }
////        LOGGER.i("initial python pass start");
//        Python py = Python.getInstance();
//        PyObject pyObject = py.getModule("plot");
////        PyObject pyObject = py.getModule("plot").callAttr("plot", inputData[0],inputData[1],
//                                                    inputData[2],inputData[3],inputData[4]);
//
//        // 将Python返回值换为Java中的Integer类型
//        Double sum = pyObject.toJava(double.class);
//        LOGGER.i("initial python get module" + pyObject.toString());

        //if select camera new activity will create, DetectorActivity
        //start new intent
        cameraButton.setOnClickListener(v -> startActivity(new Intent(MainActivity.this, DetectorActivity.class)));

        //if select camera new activity will create, DetectorActivity
        //start new intent
        btnExternalCamera.setOnClickListener(v -> startActivity(new Intent(MainActivity.this, ExternalDetectorActivity.class)));

        //when press detect
        detectButton.setOnClickListener(v -> {
            Handler handler = new Handler();

            new Thread(() -> {
                final long startTime = System.currentTimeMillis ();//SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = detector.recognizeImage(cropBitmap);
                copyResults = results;

                lastProcessingTimeMs = System.currentTimeMillis () - startTime;
//                LOGGER.i("YOLO whole time:" + lastProcessingTimeMs);
//                LOGGER.i("yolov5 result bbox size:" + results.size());
                handler.post(new Runnable() {
                    @RequiresApi(api = Build.VERSION_CODES.N)
                    @Override
                    public void run() {
                        handleResult(cropBitmap, results);

                    }
                });
            }).start();


        });


        this.sourceBitmap = Utils.getBitmapFromAsset(MainActivity.this, "kite.jpg");

        this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);

        this.imageView.setImageBitmap(cropBitmap);

        initBox();
        ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        ConfigurationInfo configurationInfo = activityManager.getDeviceConfigurationInfo();

        System.err.println(Double.parseDouble(configurationInfo.getGlEsVersion()));
        System.err.println(configurationInfo.reqGlEsVersion >= 0x30000);
        System.err.println(String.format("%X", configurationInfo.reqGlEsVersion));


    }


    private void initBox() {
        previewHeight = TF_OD_API_INPUT_SIZE;
        previewWidth = TF_OD_API_INPUT_SIZE;
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        tracker = new MultiBoxTracker(this);
        trackingOverlay = findViewById(R.id.tracking_overlay);
        LOGGER.i("mUVCCameraView MA trackingOverlay before");
        trackingOverlay.addCallback(
                canvas -> tracker.draw(canvas));
        LOGGER.i("mUVCCameraView MA trackingOverlay after");
        //only set the frame size, eg 640x640
        tracker.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation);
        LOGGER.i("mUVCCameraView MA TF_OD_API_INPUT_SIZE" + TF_OD_API_INPUT_SIZE);
        LOGGER.i("mUVCCameraView MA sensorOrientation" + sensorOrientation);
        try {
            detector =
                    YoloV5Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED,
                            TF_OD_API_INPUT_SIZE);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results) {
        final Canvas canvas = new Canvas(bitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                if (result.getDetectedClass()== 0){
                    paint.setColor(Color.BLUE);
                    paint.setStyle(Paint.Style.STROKE);
                    paint.setStrokeWidth(2.0f);
                }else{
                    paint.setColor(Color.RED);
                    paint.setStyle(Paint.Style.STROKE);
                    paint.setStrokeWidth(2.0f);
                }

                canvas.drawRect(location, paint);
//                LOGGER.i("mUVCCameraView MA result" + result.getDetectedClass());
//                cropToFrameTransform.mapRect(location);
//
//                result.setLocation(location);
//                mappedRecognitions.add(result);
            }
        }
//        tracker.trackResults(mappedRecognitions, new Random().nextInt());
//        trackingOverlay.postInvalidate();
        imageView.setImageBitmap(bitmap);

        // Feature Extraction Preprocessing
        showCropInfo(String.valueOf((int)results.size()-1));

        showInference( "yoloV5 infer time" + " "+ lastProcessingTimeMs + "ms"+
                "Bunch:"+ Bunch_No + "  " +
                "Berry:" + (results.size()-1)


        );


    }
    protected void showCropInfo(String cropInfo) {
        cropValueTextView.setText(cropInfo);
    }
    protected void showInference(String inferenceTime) {
        log_information.setText(inferenceTime);
    }
    // 1. to separate the bbox to bunch and berry
    @RequiresApi(api = Build.VERSION_CODES.N)
    public static ArrayList<Classifier.Recognition> separate_bbox_cls(List<Classifier.Recognition> results, int classID) {
        ArrayList<Classifier.Recognition> bunch_bboxes = new ArrayList<Classifier.Recognition>();

        List<Classifier.Recognition> prediction = results;
        //LOGGER.i("prediction:" + prediction);
        //1. prediction already filter out bbox conf < 0.3 (so here no need)
        //2. separate bunch and berry bbox: getDetectedClass : 0-bunch, 1-berry
        //LOGGER.i("prediction:" + prediction);
        //LOGGER.i("prediction size:" + prediction.size());
        prediction.forEach((temp) ->{
            //LOGGER.i("prediction temp:" + temp);
            if (temp.getDetectedClass() == classID) {
                bunch_bboxes.add(temp);
            }
        });

        return bunch_bboxes;
    }

    //2. find one bunch
    public ArrayList<Classifier.Recognition> find_target_bunch(Bitmap cropBitmap, ArrayList<Classifier.Recognition> arrayList) {
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

    //4. remove berry out of the selected bunch
    @RequiresApi(api = Build.VERSION_CODES.N)
    public ArrayList<Classifier.Recognition>  remove_unexpected_berry(ArrayList<Classifier.Recognition> filter_berry,
                                                                      ArrayList<Classifier.Recognition> selected_bunch) {
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

        return selected_berry;
    }

    public float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
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

    @RequiresApi(api = Build.VERSION_CODES.N)
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

    public float findLargest(float a, float b) {
        if (a > b){ return a; }
        else return b;
    }

    public float calc_area_bbox(RectF a) {
        //float area = (a.right - a.left) * (a.bottom - a.top);
        return (a.right - a.left) * (a.bottom - a.top);
    }

    public float calc_ratio(int a, int b) {
        return (float)a/b;
    }


}
