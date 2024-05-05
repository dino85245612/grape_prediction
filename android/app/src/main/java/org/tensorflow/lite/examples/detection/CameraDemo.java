
package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.hardware.usb.UsbDevice;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

//import com.chaquo.python.Python;
//import com.chaquo.python.android.AndroidPlatform;
import com.serenegiant.common.BaseActivity;
import com.serenegiant.usb.CameraDialog;
import com.serenegiant.usb.IFrameCallback;
import com.serenegiant.usb.USBMonitor;
import com.serenegiant.usb.UVCCamera;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public abstract class CameraDemo extends BaseActivity implements CameraDialog.CameraDialogParent, IFrameCallback  {
    private static final boolean DEBUG = true;	// TODO set false when production
    private static final String TAG = "CameraDemo";

    private final Object mSync = new Object();
    // for accessing USB and USB camera
    private USBMonitor mUSBMonitor;
    private UVCCamera mUVCCamera;
    private SurfaceView mUVCCameraView;
    // for open&start / stop&close camera preview
    private ImageButton mCameraButton;
    private Surface mPreviewSurface;
    private ImageView uvc_image, uvc_image2;
    private boolean isActive, isPreview;
    TextView mtvBerryNum;

    int camera_previewWidth = 0;
    int camera_previewHeight = 0;

    int getCamera_previewWidth = 0;
    int getCamera_previewHeight = 0;
    int getCamera_previewWidth_temp = 0;
    int getCamera_previewHeight_temp = 0;

    private boolean debug = false;

    int currentNumThreads = -1;
    protected Handler handler = new Handler();
    private HandlerThread handlerThread;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private int[] rgbBytes = null;
    int predictBerryNumber;
    int previousBerryNumber;
    int averagePreviousBerryNumber =0;
    int averagePredictBerryNumber;
    int zeroBerryTimes;
    List<Integer> predictBerryNumber10 = new ArrayList<Integer>();
    Handler handlerVoice = new Handler();
    protected TextToSpeech textToSpeech;
    byte[] array;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "onCreate");
        setContentView(R.layout.camerademo);
        mCameraButton = (ImageButton)findViewById(R.id.camera_button);

        mCameraButton.setOnClickListener(mOnClickListener); //no need to click the image button
//        if (mUVCCamera == null) {
//            // XXX calling CameraDialog.showDialog is necessary at only first time(only when app has no permission).
//            Log.i(TAG, "onCreate, mUVCCamera == null" );
//            CameraDialog.showDialog(CameraDemo.this);
//        } else {
//            synchronized (mSync) {
//                mUVCCamera.destroy();
//                mUVCCamera = null;
//                isActive = isPreview = false;
//            }
//        }

//        Log.i(TAG, "initial python start");
//        if (! Python.isStarted()) {
//            Python.start(new AndroidPlatform(this));
////            Log.i(TAG, "initial python enter start");
//        }
////        Log.i(TAG, "initial python pass start");

        mUVCCameraView = (SurfaceView)findViewById(R.id.camera_surface_view);
        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        mUVCCameraView.getHolder().addCallback(mSurfaceViewCallback);
//        mUVCCameraView.getLayoutParams();


        uvc_image = (ImageView)findViewById(R.id.convert_surface_view);
        mtvBerryNum = findViewById(R.id.crop_info);
        mUSBMonitor = new USBMonitor(this, mOnDeviceConnectListener);

//        getLayoutId();
//        onYOLOChosen();

        // create an object textToSpeech and adding features into it
        textToSpeech = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int i) {
                // if No error is found then only it will run
                if(i!=TextToSpeech.ERROR){
                    // To Choose language of speech
                    textToSpeech.setLanguage(Locale.JAPAN);
                }
            }
        });


    }

    protected void showCropInfo(String cropInfo) {
        mtvBerryNum.setText(cropInfo);
    }


    protected abstract int getLayoutId();

    protected abstract void onYOLOChosen();


    @Override
    protected void onStart() {
        super.onStart();
        if (DEBUG) Log.v(TAG, "onStart:");
        synchronized (mSync) {
            if (mUSBMonitor != null) {
                mUSBMonitor.register();
            }
        }
    }

    @Override
    protected void onStop() {
        if (DEBUG) Log.v(TAG, "onStop:");
        synchronized (mSync) {
            if (mUSBMonitor != null) {
                mUSBMonitor.unregister();
            }
        }
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        if (DEBUG) Log.v(TAG, "onDestroy:");
        synchronized (mSync) {
            isActive = isPreview = false;
            if (mUVCCamera != null) {
                mUVCCamera.destroy();
                mUVCCamera = null;
            }
            if (mUSBMonitor != null) {
                mUSBMonitor.destroy();
                mUSBMonitor = null;
            }
        }
        mUVCCameraView = null;
        mCameraButton = null;
        super.onDestroy();
    }

    private int buf_sz = 4096;
    private byte[] buf = new byte[buf_sz];
    Bitmap tempBitmap;


    private final IFrameCallback mIFrameCallback1 = new IFrameCallback() {
        @Override
        public void onFrame(final ByteBuffer frame) {
//            Log.v(TAG, "onFrame Trigger.");

            synchronized (mSync) {
                if (tempBitmap == null) {
//                    Log.i(TAG, "mUVCCameraView tempBitmap width" + getCamera_previewWidth);
//                    Log.i(TAG, "mUVCCameraView tempBitmap height" + getCamera_previewHeight);
//                    Log.i(TAG, "mUVCCameraView .get LayoutParams()" + mUVCCameraView.getLayoutParams().toString());
//                    Log.i(TAG, "mUVCCameraView .get Height()" + mUVCCameraView.getHeight());
//                    Log.i(TAG, "mUVCCameraView .get Width()" + mUVCCameraView.getWidth());
                    tempBitmap = Bitmap.createBitmap(getCamera_previewWidth, getCamera_previewHeight, Bitmap.Config.ARGB_8888);

                }
                frame.clear();
//                array = frame.array();
                tempBitmap.copyPixelsFromBuffer(frame);
            }

            uvc_image.post(mUpdateImageTask);

//            array = frame.array();


            predictBerryNumber = processImage();
//            Log.i("CameraDemo ", "predictBerryNumber: " + (int)predictBerryNumber);
            //showCropInfo(String.valueOf((int)predictBerryNumber));

            previousBerryNumber = predictBerryNumber;
//            Log.i("predictBerryNumber", "predictBerryNumber: " + predictBerryNumber);
//            Log.i("predictBerryNumber", "previousBerryNumber: " + previousBerryNumber);

            if(predictBerryNumber != 0) {
                predictBerryNumber10.add(predictBerryNumber);
                zeroBerryTimes = 0;
            }else{
                if (previousBerryNumber == 0 && predictBerryNumber == 0){
                    ++zeroBerryTimes;
                }
            }
//            Log.i("predictBerryNumber", "zeroBerryTimes: >>>>>>>>>" + zeroBerryTimes);
//
//            Log.i("predictBerryNumber10", "predictBerryNumber10: " + predictBerryNumber10.size());
            if (predictBerryNumber10.size() == 3 && averagePreviousBerryNumber == 0) {//first time
                int sum = sum(predictBerryNumber10);
                averagePredictBerryNumber = (int) sum/3;
                handlerVoice.postDelayed(mVoiceRunnable, 10);
                averagePreviousBerryNumber = averagePredictBerryNumber;
                showCropInfo(String.valueOf((int)averagePredictBerryNumber));
            }
            else if (predictBerryNumber10.size() == 20) {
                // get the sum of the elements in the list
                int sum = sum(predictBerryNumber10);
                averagePredictBerryNumber = (int) sum/20;

                int difference = averagePredictBerryNumber - averagePreviousBerryNumber;
//                Log.i("predictBerryNumber10", "difference: " + difference);
                if (difference == 0 && averagePredictBerryNumber != 0){ //nth time
                    handlerVoice.postDelayed(mVoiceRunnable, 20000);
                    showCropInfo(String.valueOf((int)averagePredictBerryNumber));
                }
                else if ((difference > 2 || difference < -2) && averagePredictBerryNumber != 0){ //nth time
                    handlerVoice.postDelayed(mVoiceRunnable, 20000);
                    showCropInfo(String.valueOf((int)averagePredictBerryNumber));
                }
                else{
                    handlerVoice.postDelayed(mVoiceRunnable, 3000);
                    showCropInfo(String.valueOf((int)averagePredictBerryNumber));
                }
                averagePreviousBerryNumber = averagePredictBerryNumber;
                predictBerryNumber10.removeAll(predictBerryNumber10);
                showCropInfo(String.valueOf((int)averagePredictBerryNumber));

            }
            else{
                if (zeroBerryTimes == 20){
//                    Log.i("predictBerryNumber10", "zeroBerryTimes >>>>>>>>>hit: " + zeroBerryTimes);
                    handlerVoice.removeCallbacks(mVoiceRunnable);
                    zeroBerryTimes = 0;
                    averagePreviousBerryNumber = 0;
                    predictBerryNumber10.removeAll(predictBerryNumber10);
                    showCropInfo(String.valueOf((int)averagePreviousBerryNumber));
                }
            }
        }
    };

    protected abstract int processImage();

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
            textToSpeech.speak(String.valueOf(averagePredictBerryNumber), TextToSpeech.QUEUE_FLUSH,null);
//      Log.i("predictBerryNumber", "handler after voice: " + averagePredictBerryNumber);
        }
    };

    private Runnable mUpdateImageTask = new Runnable() {
        @Override
        public void run() {
            synchronized (mSync) {
                uvc_image.setImageBitmap(tempBitmap);
            }
        }
    };

    private final View.OnClickListener mOnClickListener = new View.OnClickListener() {
        @Override
        public void onClick(final View view) {
            if (mUVCCamera == null) {
                // XXX calling CameraDialog.showDialog is necessary at only first time(only when app has no permission).
                CameraDialog.showDialog(CameraDemo.this);
            } else {
                synchronized (mSync) {
                    mUVCCamera.destroy();
                    mUVCCamera = null;
                    isActive = isPreview = false;
                }
            }
        }
    };

    private final USBMonitor.OnDeviceConnectListener mOnDeviceConnectListener = new USBMonitor.OnDeviceConnectListener() {
        @Override
        public void onAttach(final UsbDevice device) {
            if (DEBUG) Log.v(TAG, "onAttach:");
            Toast.makeText(CameraDemo.this, "USB_DEVICE_ATTACHED", Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onDetach(final UsbDevice device) {
            if (DEBUG) Log.v(TAG, "onDettach:");
            Toast.makeText(CameraDemo.this, "USB_DEVICE_DETACHED", Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onConnect(final UsbDevice device, final USBMonitor.UsbControlBlock ctrlBlock, final boolean createNew) {
            Log.i(TAG, "onConnect ");
            if (DEBUG) Log.v(TAG, "onConnect:");
            synchronized (mSync) {
                if (mUVCCamera != null) {
                    mUVCCamera.destroy();
                }
                isActive = isPreview = false;
            }
            queueEvent(new Runnable() {
                @Override
                public void run() {
                    synchronized (mSync) {
                        final UVCCamera camera = new UVCCamera();
                        camera.open(ctrlBlock);
                        if (DEBUG) Log.i(TAG, "supportedSize:" + camera.getSupportedSize());
//                        Log.i(TAG, "mUVCCameraView getSupportedSize:" + camera.getSupportedSize());
//                        Log.i(TAG, "mUVCCameraView getSupportedSize:" + camera.getSupportedSize());
                        String supportSize = camera.getSupportedSize();
//                        Log.i(TAG, "mUVCCameraView supportSizelength():" + supportSize);

                        if (supportSize.contains("640x480")){
                            camera_previewWidth = 640;
                            camera_previewHeight = 480;
                            getCamera_previewWidth = 640;//camera.getPreviewSize().width;//UVCCamera.DEFAULT_PREVIEW_WIDTH;
                            getCamera_previewHeight = 480;//camera.getPreviewSize().height;//UVCCamera.DEFAULT_PREVIEW_HEIGHT;
//                            Log.i(TAG, "mUVCCameraView supportSize contains: 640x480" + supportSize.contains("640x480"));
                        }else if (supportSize.contains("1280x720")){
                            camera_previewWidth = 1280;
                            camera_previewHeight = 720;
                            getCamera_previewWidth = 1280;//camera.getPreviewSize().width;//UVCCamera.DEFAULT_PREVIEW_WIDTH;
                            getCamera_previewHeight = 720;//camera.getPreviewSize().height;//UVCCamera.DEFAULT_PREVIEW_HEIGHT;
//                            Log.i(TAG, "mUVCCameraView supportSize contains: 1280x720" + supportSize.contains("1280x720"));
                        }
//

                        try {
                            camera.setPreviewSize(camera_previewWidth, camera_previewHeight, UVCCamera.FRAME_FORMAT_MJPEG);
//                            Log.i(TAG, "mUVCCameraView FRAME_FORMAT_MJPEG");
                        } catch (final IllegalArgumentException e) {
                            try {
                                // fallback to YUV mode
                                camera.setPreviewSize(camera_previewWidth, camera_previewHeight, UVCCamera.FRAME_FORMAT_YUYV);
//                                Log.i(TAG, "mUVCCameraView catch FRAME_FORMAT_YUYV" );
                            } catch (final IllegalArgumentException e1) {
                                camera.destroy();
//                                Log.i(TAG, "mUVCCameraView catch try" );
                                return;
                            }
                        }
//                        Log.i(TAG, "mUVCCameraView getPreviewSize width:" + camera.getPreviewSize().width);
//                        Log.i(TAG, "mUVCCameraView getPreviewSize height:" + camera.getPreviewSize().height);
                        getCamera_previewHeight_temp = camera.getPreviewSize().height;//UVCCamera.DEFAULT_PREVIEW_HEIGHT;
                        getCamera_previewWidth_temp = camera.getPreviewSize().width;//UVCCamera.DEFAULT_PREVIEW_WIDTH;

                        onYOLOChosen();

                        // here is start displaying the image from the camera
                        mPreviewSurface = mUVCCameraView.getHolder().getSurface();
//
                        if (mPreviewSurface != null) {
                            isActive = true;
                            camera.setPreviewDisplay(mPreviewSurface); // The Surface has been created, now tell the camera where to draw the preview.
                            camera.startPreview();
                            isPreview = true;
                        }



                        synchronized (mSync) {
                            mUVCCamera = camera; //frame size 640*480
//                            Log.i(TAG, "mUVCCameraView mSync mUVCCamera.getPreviewSize():" + mUVCCamera.getPreviewSize());
                            mUVCCamera.setFrameCallback(mIFrameCallback1, UVCCamera.PIXEL_FORMAT_RGBX);

                        }
                    }
                }
            }, 0);
        }

        @Override
        public void onDisconnect(final UsbDevice device, final USBMonitor.UsbControlBlock ctrlBlock) {
            if (DEBUG) Log.v(TAG, "onDisconnect:");
            // XXX you should check whether the comming device equal to camera device that currently using
            queueEvent(new Runnable() {
                @Override
                public void run() {
                    synchronized (mSync) {
                        if (mUVCCamera != null) {
                            mUVCCamera.close();
                            if (mPreviewSurface != null) {
                                mPreviewSurface.release();
                                mPreviewSurface = null;
                            }
                            isActive = isPreview = false;
                        }
                    }
                }
            }, 0);
        }



        @Override
        public void onCancel(final UsbDevice device) {
        }
    };

    /**
     * to access from CameraDialog
     * @return
     */
    @Override
    public USBMonitor getUSBMonitor() {
        return mUSBMonitor;
    }

    @Override
    public void onDialogResult(boolean canceled) {
        if (canceled) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    // FIXME
                }
            }, 0);
        }
    }

    private final SurfaceHolder.Callback mSurfaceViewCallback = new SurfaceHolder.Callback() {
        @Override
        public void surfaceCreated(final SurfaceHolder holder) {
            if (DEBUG) Log.v(TAG, "surfaceCreated:");
        }

        @Override
        public void surfaceChanged(final SurfaceHolder holder, final int format, final int width, final int height) {
            if ((width == 0) || (height == 0)) return;
            if (DEBUG) Log.v(TAG, "surfaceChanged:");
            mPreviewSurface = holder.getSurface();
            synchronized (mSync) {
                if (isActive && !isPreview && (mUVCCamera != null)) {
                    mUVCCamera.setPreviewDisplay(mPreviewSurface);
                    mUVCCamera.startPreview();
                    isPreview = true;
                }
            }
        }

        @Override
        public void surfaceDestroyed(final SurfaceHolder holder) {
            if (DEBUG) Log.v(TAG, "surfaceDestroyed:");
            synchronized (mSync) {
                if (mUVCCamera != null) {
                    mUVCCamera.stopPreview();
                }
                isPreview = false;
            }
            mPreviewSurface = null;
        }
    };

    @Override
    public void onFrame(ByteBuffer byteBuffer) {
        Log.v(TAG, "onFrame:");
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

    public boolean isDebug() {
        return debug;
    }

    protected Bitmap getRgbBytes() {
//        imageConverter.run();
//        int width = tempBitmap.getWidth();
//        int height = tempBitmap.getHeight();
//        int[] pixels = new int[width * height];
//
//        // 8Bit=>16Bit配列へ変換
//        short[] buf = new short[array.length / 2];
//        ByteBuffer bb = ByteBuffer.wrap(array);
//        for (int i = 0; i < buf.length; i++) {
//            buf[i] = bb.getShort();
//        }
//        // BitMapへ設定
//        for(int y = 0; y < buf.length/width; y+=1){
//            for(int x = 0; x < width; x += 1){
//                int c = x + y * width;
//                int redp = (buf[c] & 0xf800) >> 8 | (buf[c] & 0xe000) >> 13;
//                int greenp = (buf[c] & 0x07e0) >> 3 | (buf[c] & 0x0600) >> 9;
//                int bluep = (buf[c] & 0x001f) << 3 | (buf[c] & 0x001c) >> 2;
//                pixels[c] = Color.argb(255, redp, greenp, bluep );
//                //Log.d(TAG, "r:" + redp + ",g:" + greenp + ",b:" + bluep );
//            }
//        }
//        uvc_image2 = (ImageView)findViewById(R.id.convert_surface_view2);



//        tempBitmap = scaleBitmap(tempBitmap, mUVCCameraView.getWidth(), mUVCCameraView.getHeight());
////        tempBitmap = Bitmap.createScaledBitmap(b, mUVCCameraView.getWidth(), mUVCCameraView.getHeight(), false);
//        uvc_image2.setImageBitmap(tempBitmap);
        return tempBitmap;
    }

    public static Bitmap scaleBitmap(Bitmap bitmapToScale, float newWidth, float newHeight) {
        if(bitmapToScale == null)
            return null;
        //get the original width and height
        int width = bitmapToScale.getWidth();
        int height = bitmapToScale.getHeight();
        // create a matrix for the manipulation
        Matrix matrix = new Matrix();

        // resize the bit map
        matrix.postScale(newWidth / width, newHeight / height);

        // recreate the new Bitmap and set it back
        return Bitmap.createBitmap(bitmapToScale, 0, 0, bitmapToScale.getWidth(), bitmapToScale.getHeight(), matrix, true);  }

    protected int getPreviewHeight() {
//        imageConverter.run();

        return camera_previewHeight;
    }

    protected int getPreviewWidth() {
//        imageConverter.run();
        return camera_previewWidth;
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

//    protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

    protected void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }
}
