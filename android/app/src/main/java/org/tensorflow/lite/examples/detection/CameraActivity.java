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

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.google.android.material.bottomsheet.BottomSheetBehavior;

import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;


public abstract class CameraActivity extends AppCompatActivity implements OnImageAvailableListener,
        SpeechRecognitionListener.OnCommandReceivedListener,
        Camera.PreviewCallback,
//        CompoundButton.OnCheckedChangeListener,
        View.OnClickListener
{
  private static final Logger LOGGER = new Logger();

  private static final int PERMISSIONS_REQUEST = 1;

  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
  private static final String ASSET_PATH = "";
  protected int previewWidth = 0;
  protected int previewHeight = 0;
  private boolean debug = false;
  protected Handler handler;
  private HandlerThread handlerThread;
  private boolean useCamera2API;
  private boolean isProcessingFrame = false;
  private byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;
  protected int defaultModelIndex = 0;
  protected int defaultDeviceIndex = 1;
  private Runnable postInferenceCallback;
  private Runnable imageConverter;
  protected ArrayList<String> modelStrings = new ArrayList<String>();

  private long lastProcessingTimeMs;

  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  protected TextView frameValueTextView, cropValueTextView, inferenceTimeTextView, minValueTextView;
  protected TextView maxValueTextView, processTextView;
  protected ImageView bottomSheetArrowImageView;
  private ImageView plusImageView, minusImageView;
  protected ListView deviceView;
  protected TextView threadsTextView;
  protected ListView modelView;
  protected TextToSpeech textToSpeech;
  private Button saveButton;
  private TextView tvSaveFolderNameResult;
  private EditText etFolderName;
  /** Current indices of device and model. */
  int currentDevice = -1;
  int currentModel = -1;
  int currentNumThreads = -1;
  int predictBerryNumber;
  int previousBerryNumber;

  int averagePredictBerryNumber;

  boolean get_result = false;

  private static final int DESIRED_FRAME_RATE = 5; // Set this value to the desired frame rate
  int frameNo = 4;
//  int frameNo = 2; //]j/m google glass
  int accumFrame = 0;
  int notifyTimes = 3;
  int maximum = 0;
  int minimum = 0;
  int average = 0;
  int zeroTimes = 0;
  int Process_stage = 0;
  List<Integer> accumDisplayResult = new ArrayList<Integer>();

  int skip_notifyTimes = 0;

  List<Integer> accumPredictBerryNumber = new ArrayList<Integer>();
  Handler handlerVoice = new Handler();

  ArrayList<String> deviceStrings = new ArrayList<String>();
  private final String BunchName = null;
  private String bunchId = null;
  private int berryNum = 0;
  private Button buttonResnet, buttonRFR, selectedButton;
  private Switch switchQRVoice;
  private static final int SPEECH_REQUEST_CODE = 0;
  int logBerryNum;
  private String rqResult;
  private SpeechRecognizer speechRecognizer;
  private Intent speechRecognizerIntent;
  private SpeechRecognitionListener speechRecognitionListener;

  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    setContentView(R.layout.tfe_od_activity_camera);
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);
    getSupportActionBar().setDisplayShowTitleEnabled(false);

    if (hasPermission()) {
      setFragment();
    } else {
      requestPermission();
    }
    BunchDetail bunchDetail = new BunchDetail();
    bunchDetail.setBunchId(bunchId); // replace bunchId with the scanned information
    bunchDetail.setBunchName(BunchName); // replace bunchName with the scanned information
    bunchDetail.setberryNumN(berryNum); // set the current average we stored earlier

    threadsTextView = findViewById(R.id.threads);
    currentNumThreads = Integer.parseInt(threadsTextView.getText().toString().trim());
    plusImageView = findViewById(R.id.plus);
    minusImageView = findViewById(R.id.minus);
    deviceView = findViewById(R.id.device_list);
    deviceStrings.add("CPU");
    deviceStrings.add("GPU");
    deviceStrings.add("NNAPI");
    deviceView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
    ArrayAdapter<String> deviceAdapter =
            new ArrayAdapter<>(
                    CameraActivity.this , R.layout.deviceview_row, R.id.deviceview_row_text, deviceStrings);
    deviceView.setAdapter(deviceAdapter);
    deviceView.setItemChecked(defaultDeviceIndex, true);
    currentDevice = defaultDeviceIndex;
    deviceView.setOnItemClickListener(
            new AdapterView.OnItemClickListener() {
              @Override
              public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
//                updateActiveModel();
              }
            });

    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    modelView = findViewById((R.id.model_list));

    modelStrings = getModelStrings(getAssets(), ASSET_PATH);
    modelView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
    ArrayAdapter<String> modelAdapter =
            new ArrayAdapter<>(
                    CameraActivity.this , R.layout.listview_row, R.id.listview_row_text, modelStrings);
    modelView.setAdapter(modelAdapter);
    modelView.setItemChecked(defaultModelIndex, true); //set this model to check
    currentModel = defaultModelIndex;
    modelView.setOnItemClickListener(
            new AdapterView.OnItemClickListener() {
              @Override
              public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
//                updateActiveModel();
              }
            });

    frameValueTextView = findViewById(R.id.frame_info);
    cropValueTextView = findViewById(R.id.crop_info);
//    inferenceTimeTextView = findViewById(R.id.inference_info);
    minValueTextView = findViewById(R.id.crop_info_detail);
    maxValueTextView = findViewById(R.id.crop_info_detail2);
    processTextView = findViewById(R.id.crop_info_ing);

    // create an object textToSpeech and adding features into it
    textToSpeech = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
      @Override
      public void onInit(int i) {

        // if No error is found then only it will run
        if(i!=TextToSpeech.ERROR){
          // To Choose language of speech
          textToSpeech.setLanguage(Locale.JAPANESE);
        }
      }
    });

    // Find the button and set the click listener
    Button qrCodeButton = findViewById(R.id.qr_code_button);
    qrCodeButton.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        onQRCodeButtonClick();
      }
    });

    // This is where you might want to start listening for voice commands
    // For example, you might have a voice command button in your UI
    switchQRVoice = findViewById(R.id.qr_code_voice);
    switchQRVoice.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
      @Override
      public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        if (isChecked) {
          // The switch is enabled/checked, start voice recognition
          startListening();
        } else {
          // The switch is disabled/unchecked, stop voice recognition
          stopListening();
        }
      }
    });

    buttonResnet = findViewById(R.id.buttonResnet);
    buttonRFR = findViewById(R.id.buttonRandomForest);

    // Set initial selected button to "Resnet"
    selectedButton = buttonResnet;
    buttonResnet.setBackgroundColor(getResources().getColor(R.color.button_selected));

    buttonResnet.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        // Switch to "Resnet" mode
        selectedButton.setBackgroundColor(getResources().getColor(R.color.button_normal));
        selectedButton = buttonResnet;
        buttonResnet.setBackgroundColor(getResources().getColor(R.color.button_selected));
      }
    });

    buttonRFR.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        // Switch to "Random Forest" mode
        selectedButton.setBackgroundColor(getResources().getColor(R.color.button_normal));
        selectedButton = buttonRFR;
        buttonRFR.setBackgroundColor(getResources().getColor(R.color.button_selected));
      }
    });

    saveButton = findViewById(R.id.buttonSave);
    etFolderName = findViewById(R.id.editTextTextPersonName);
    tvSaveFolderNameResult = (TextView)findViewById(R.id.tvSaveFolderName);
    SharedPreferences sharedPref = getSharedPreferences("Grape", Context.MODE_PRIVATE);
    SharedPreferences.Editor editor = sharedPref.edit();
    editor.putString("inputFolderName", "noInput");
    editor.apply();

    saveButton.setOnClickListener(new View.OnClickListener() {
      public void onClick(View view) {
        // get text from EditText name view
        String inputFolderName = etFolderName.getText().toString();
        tvSaveFolderNameResult.setText(" "+ inputFolderName);

        // Save to shared preferences
        SharedPreferences sharedPref = getSharedPreferences("Grape", Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPref.edit();
        editor.putString("inputFolderName", inputFolderName);
        editor.apply();

      }
    });

  }

  private void startListening() {
    LOGGER.i("voices: enter startListening ");
    // Initialize the SpeechRecognitionListener
    speechRecognitionListener = new SpeechRecognitionListener();
    speechRecognitionListener.setOnCommandReceivedListener(this);

    // Initialize the SpeechRecognizer
    LOGGER.i("voices: enter startListening: Initialize the SpeechRecognizer ");
    speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);

    // Initialize the SpeechRecognizerIntent
    LOGGER.i("voices: enter startListening: Initialize the SpeechRecognizerIntent ");
    Intent speechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
    speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
    speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, this.getPackageName());

    // Start listening
    LOGGER.i("voices: enter startListening: Start listening ");
    speechRecognizer.setRecognitionListener(speechRecognitionListener);
    speechRecognizer.startListening(speechRecognizerIntent);
  }

  private void stopListening() {
    // Stop listening
    if (speechRecognizer != null) {
      speechRecognizer.stopListening();
      speechRecognizer.destroy();
    }
  }


  @Override
  public void onQRCodeCommand() {
    onQRCodeButtonClick();
  }

  protected void displaySpeechRecognizer(){
    Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
    intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
    startActivityForResult(intent, SPEECH_REQUEST_CODE);
  }

  private void onQRCodeButtonClick() {
    LOGGER.i("voices: start onQRCodeButtonClick %%%%%% ");
    // Start a new activity to scan and detect QR code
    Intent intent = new Intent(CameraActivity.this, QRCodeScannerActivity.class);
    intent.putExtra("average", logBerryNum); // Pass the average value
    startActivityForResult(intent, 1); // Start the activity and expect a result

  }

  @Override
  protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
    LOGGER.i("voices: requestCode: " + requestCode);
    LOGGER.i("voices: requestCode: " + requestCode);
    // Check if the result is from QRCodeScannerActivity
    if (requestCode == 1) {
      // Check if the result is successful
      if (resultCode == RESULT_OK) {
        // Show a Toast message to the user
        rqResult = "QR scanned successfully";
        Toast.makeText(this, "QR scanned successfully", Toast.LENGTH_LONG).show();
      } else if (resultCode == RESULT_CANCELED) {
        // The task failed, show a different Toast message
        rqResult = "QR scanned failed";
        Toast.makeText(this, "QR scanned failed", Toast.LENGTH_LONG).show();
      }
      handlerVoice.postDelayed(mVoiceRunnable3, 10);
    }
    else if (requestCode == SPEECH_REQUEST_CODE) {
      LOGGER.i("voices: enter requestCode == 0 ");
      List<String> results = data.getStringArrayListExtra(
              RecognizerIntent.EXTRA_RESULTS);
      String spokenText = results.get(0);
      LOGGER.i("voices: spokenText: " + spokenText);
      if (spokenText.contains("QR")) {
        Toast.makeText(this, "QR scanned QR", Toast.LENGTH_LONG).show();
        LOGGER.i("voices: enter onQRCodeButtonClick %%%%%% ");
        onQRCodeButtonClick();
        LOGGER.i("voices: end onQRCodeButtonClick %%%%%% ");
      }
      else{
        Toast.makeText(this, "QR scanned not QR", Toast.LENGTH_LONG).show();
      }
    }
  }


  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  protected ArrayList<String> getModelStrings(AssetManager mgr, String path){ //not necessary ????
    ArrayList<String> res = new ArrayList<String>();
    try {
      String[] files = mgr.list(path);
      for (String file : files) {
        String[] splits = file.split("\\.");
        if (splits[splits.length - 1].equals("tflite")) {
          res.add(file);
        }
      }

    }
    catch (IOException e){
      System.err.println("getModelStrings: " + e.getMessage());
    }
    return res;
  }

  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }

  protected int getLuminanceStride() {
    return yRowStride;
  }

  protected byte[] getLuminance() {
    return yuvBytes[0];
  }

  /** Callback for android.hardware.Camera API */
  @Override
  public void onPreviewFrame(final byte[] bytes, final Camera camera) {
    if (isProcessingFrame) {
      LOGGER.w("Dropping frame!");
      return;
    }

    try {
      // Initialize the storage bitmaps once when the resolution is known.
      if (rgbBytes == null) {
        Camera.Size previewSize = camera.getParameters().getPreviewSize();
        previewHeight = previewSize.height;
        previewWidth = previewSize.width;
        rgbBytes = new int[previewWidth * previewHeight];
        onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
      }
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      return;
    }

    isProcessingFrame = true;
    yuvBytes[0] = bytes;
    yRowStride = previewWidth;

    imageConverter =
            new Runnable() {
              @Override
              public void run() {
                ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
              }
            };

    postInferenceCallback =
            new Runnable() {
              @Override
              public void run() {
                camera.addCallbackBuffer(bytes);
                isProcessingFrame = false;
              }
            };
    predictBerryNumber = processImage();
  }

  /** Callback for Camera2 API */
  @Override
  public void onImageAvailable(final ImageReader reader) {
    // We need to control the frame rate to limit it to our desired value
    long currentTimeMs = System.currentTimeMillis();
    if (currentTimeMs - lastProcessingTimeMs < (1000 / DESIRED_FRAME_RATE)) {
      Image imageToClose = reader.acquireLatestImage();
      if (imageToClose != null) {
        imageToClose.close(); // Close the image before returning
      }
      return; // Skip this frame to limit the frame rate
    }
    lastProcessingTimeMs = currentTimeMs;

    // We need wait until we have some size from onPreviewSizeChosen
    if (previewWidth == 0 || previewHeight == 0) {
      return;
    }
    if (rgbBytes == null) {
      rgbBytes = new int[previewWidth * previewHeight];
    }
    try {
      final Image image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (isProcessingFrame) {
        image.close();
        return;
      }


      isProcessingFrame = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);
      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      imageConverter =
              new Runnable() {
                @Override
                public void run() {
                  ImageUtils.convertYUV420ToARGB8888(
                          yuvBytes[0],
                          yuvBytes[1],
                          yuvBytes[2],
                          previewWidth,
                          previewHeight,
                          yRowStride,
                          uvRowStride,
                          uvPixelStride,
                          rgbBytes);
                }
              };

      postInferenceCallback =
              new Runnable() {
                @Override
                public void run() {
                  image.close();
                  isProcessingFrame = false;
                }
              };

//      LOGGER.i("processImage: new -----------START------------Process_stage " + Process_stage);
      int diffCompare = 0;
      predictBerryNumber = processImage();
      if (predictBerryNumber == -999){ // mean not computing detection, so get the result
        get_result = false;
      }else if(predictBerryNumber == -777){ // bunch location is out of frame
        get_result = false;
//        accumPredictBerryNumber.removeAll(accumPredictBerryNumber);
//        accumDisplayResult.removeAll(accumDisplayResult);
//        showCropInfo(String.valueOf((int) 0));
//        showCropInfo2(String.valueOf((int) 0));
//        showCropInfo3(String.valueOf((int) 0));
//        showProcess(String.valueOf("No"));
//        previousBerryNumber = 0;
      }else if(predictBerryNumber == -333){ // rest after done detection
        get_result = false;
        Process_stage = 0;
      }else if(predictBerryNumber == 0){ // no detection
        get_result = false;
//        notifyTimes = 0;
        accumFrame = 0;
        accumPredictBerryNumber.removeAll(accumPredictBerryNumber);
        accumDisplayResult.removeAll(accumDisplayResult);
        showCropInfo(String.valueOf((int) 0));
        showCropInfo2(String.valueOf((int) 0));
        showCropInfo3(String.valueOf((int) 0));
        showProcess(String.valueOf("No"));
        previousBerryNumber = 0;
      }else{// out of !-999 and -777 and 0
        get_result = true;
        accumDisplayResult.add(predictBerryNumber);
        showProcess(String.valueOf("Yes"));
      }
//      LOGGER.i("result: -------------------------------------------------------------------------------------------------------------");
//      LOGGER.i("result: CameraActivity - predictBerryNumber = " + predictBerryNumber);
//      LOGGER.i("result: CameraActivity - result accumDisplayResult.size()= " + accumDisplayResult.size());


      if(accumDisplayResult.size() >= frameNo){
        minimum = find_smallest(accumDisplayResult);
        maximum = find_largest(accumDisplayResult);
        average = (int) sum(accumDisplayResult) / accumDisplayResult.size();
        logBerryNum = average;
        averagePredictBerryNumber = average;
        diffCompare = Math.abs(averagePredictBerryNumber - previousBerryNumber);
        LOGGER.i("result: CameraActivity - result diffCompare= " + diffCompare);
        if (diffCompare > 3){
          handlerVoice.postDelayed(mVoiceRunnable, 100);
          showCropInfo(String.valueOf((int) average));
          showCropInfo2(String.valueOf((int) minimum));
          showCropInfo3(String.valueOf((int) maximum));
        }else{
          skip_notifyTimes += 1;
//          LOGGER.i("result: CameraActivity - result skip_notifyTimes = " + skip_notifyTimes);
          if (skip_notifyTimes >= notifyTimes){
            handlerVoice.postDelayed(mVoiceRunnable, 100);
            showCropInfo(String.valueOf((int) average));
            showCropInfo2(String.valueOf((int) minimum));
            showCropInfo3(String.valueOf((int) maximum));
            skip_notifyTimes=0;
//            LOGGER.i("result: CameraActivity - result averagePredictBerryNumber = " + averagePredictBerryNumber);
          }
        }
        accumDisplayResult.removeAll(accumDisplayResult);
        showProcess(String.valueOf("Done"));
        previousBerryNumber = average;
      }


    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }
    Trace.endSection();
  }

  protected int find_largest(List<Integer> accum){
    int largest = 0;
      for (int i = 0; i < accum.size(); i++){
        if (i == 0){
          largest = accum.get(i);
        }
        else{
          if (accum.get(i) > largest){
            largest = accum.get(i);
          }
        }
      }
    return largest;
  };

  protected int find_smallest(List<Integer> accum){
    int smallest = 0;
    for (int i = 0; i < accum.size(); i++){
      if (i == 0){
        smallest = accum.get(i);
      }
      else{
        if (accum.get(i) < smallest){
          smallest = accum.get(i);
        }
      }
    }
    return smallest;
  };


  protected abstract void clearBbox();

  protected abstract boolean getBunchIdea();



  public static int sum(List<Integer> list) {
    int sum = 0;
    for (int i: list) {
      sum += i;
    }
    return sum;
  }

  private Runnable mVoiceRunnable = new Runnable() {
    @Override
    public void run() {
      textToSpeech.speak(String.valueOf(averagePredictBerryNumber),TextToSpeech.QUEUE_ADD,null);
//      Log.i("predictBerryNumber", "handler after voice: " + averagePredictBerryNumber);
    }
  };



  private Runnable mVoiceRunnable2 = new Runnable() {
    @Override
    public void run() {
      textToSpeech.speak(String.valueOf(predictBerryNumber),TextToSpeech.QUEUE_FLUSH,null);
//      Log.i("predictBerryNumber", "handler after voice: " + averagePredictBerryNumber);
    }
  };

  private Runnable mVoiceRunnable3 = new Runnable() {
    @Override
    public void run() {
      textToSpeech.speak(String.valueOf(rqResult),TextToSpeech.QUEUE_FLUSH,null);
    }
  };

  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
    // other code...
    if (speechRecognizer != null) {
      speechRecognizer.startListening(speechRecognizerIntent);
    }
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }

    super.onPause();

    // other code...
    if (speechRecognizer != null) {
      speechRecognizer.stopListening();
    }
  }

  @Override
  public synchronized void onStop() {
    LOGGER.d("onStop " + this);
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);
    super.onDestroy();
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }

  @Override
  public void onRequestPermissionsResult(
          final int requestCode, final String[] permissions, final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        setFragment();
      } else {
        requestPermission();
      }
    }
  }

  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        Toast.makeText(
                CameraActivity.this,
                "Camera permission is required for this demo",
                Toast.LENGTH_LONG)
                .show();
      }
      requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
  }

  // Returns true if the device supports the required hardware level, or better.
  private boolean isHardwareLevelSupported(
          CameraCharacteristics characteristics, int requiredLevel) {
    int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
      return requiredLevel == deviceLevel;
    }
    // deviceLevel is not LEGACY, can use numerical sort
    return requiredLevel <= deviceLevel;
  }

  private String chooseCamera() {
    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    try {
      for (final String cameraId : manager.getCameraIdList()) {
        final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

        // We don't use a front facing camera in this sample.
        final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
          continue;
        }

        final StreamConfigurationMap map =
                characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        if (map == null) {
          continue;
        }

        // Fallback to camera1 API for internal cameras that don't have full support.
        // This should help with legacy situations where using the camera2 API causes
        // distorted or otherwise broken previews.
        useCamera2API =
                (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                        || isHardwareLevelSupported(
                        characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
        LOGGER.i("Camera API lv2?: %s", useCamera2API);

        int[] afAvailableModes = characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES);
        LOGGER.i("Camera API lv2?: afAvailableModes", afAvailableModes);

        boolean macroSupported = false;
        for (int mode : afAvailableModes) {
          if (mode == CaptureRequest.CONTROL_AF_MODE_MACRO) {
            macroSupported = true;
            LOGGER.i("Camera API lv2?: mode macroSupported = true", mode);
            break;
          }
        }





        return cameraId;
      }
    } catch (CameraAccessException e) {
      LOGGER.e(e, "Not allowed to access camera");
    }

    return null;
  }

  protected void setFragment() {
    String cameraId = chooseCamera();

    Fragment fragment;
    if (useCamera2API) {
      CameraConnectionFragment camera2Fragment =
              CameraConnectionFragment.newInstance(
                      new CameraConnectionFragment.ConnectionCallback() {
                        @Override
                        public void onPreviewSizeChosen(final Size size, final int rotation) {
                          previewHeight = size.getHeight();
                          previewWidth = size.getWidth();
                          CameraActivity.this.onPreviewSizeChosen(size, rotation);
//                          LOGGER.i("processImage previewHeight: " + previewHeight + "processImage previewWidth: " + previewWidth );
                        }
                      },
                      this,
                      getLayoutId(),
                      getDesiredPreviewFrameSize());

      camera2Fragment.setCamera(cameraId);
      fragment = camera2Fragment;
    } else {
      fragment =
              new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
    }

    getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
  }



  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }

  public boolean isDebug() {
    return debug;
  }

  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
//      LOGGER.i("processImage predictBerryNumber:   postInferenceCallback: ");
      postInferenceCallback.run();
    }
  }

  protected int getScreenOrientation() {
    switch (getWindowManager().getDefaultDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }


  @Override
  public void onClick(View v) {
    if (v.getId() == R.id.plus) {
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads >= 9) return;
      numThreads++;
      threadsTextView.setText(String.valueOf(numThreads));
      setNumThreads(numThreads);
    } else if (v.getId() == R.id.minus) {
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads == 1) {
        return;
      }
      numThreads--;
      threadsTextView.setText(String.valueOf(numThreads));
      setNumThreads(numThreads);
    }
  }

  protected void showFrameInfo(String frameInfo) {
    frameValueTextView.setText(frameInfo);
  }

  protected void showCropInfo(String cropInfo) {
    cropValueTextView.setText(cropInfo);
  }

  protected void showCropInfo2(String cropInfo) {
    minValueTextView.setText(cropInfo);
  }

  protected void showCropInfo3(String cropInfo) {
    maxValueTextView.setText(cropInfo);
  }

  protected void showProcess(String cropInfo) {
    processTextView .setText(cropInfo);
  }

  protected void showInference(String inferenceTime) {
    inferenceTimeTextView.setText(inferenceTime);
  }

  //  protected abstract void updateActiveModel();
//  protected abstract boolean getBunchIdea();

  protected abstract int processImage();

  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

  protected abstract int getLayoutId();

  protected abstract Size getDesiredPreviewFrameSize();

  protected abstract void setNumThreads(int numThreads);

  protected abstract void setUseNNAPI(boolean isChecked);
}
