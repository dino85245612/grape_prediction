
package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.hardware.usb.UsbDevice;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import com.serenegiant.common.BaseActivity;
import com.serenegiant.usb.CameraDialog;
import com.serenegiant.usb.IFrameCallback;
import com.serenegiant.usb.USBMonitor;
import com.serenegiant.usb.UVCCamera;

import java.nio.ByteBuffer;

public class CameraDemoCopy extends BaseActivity implements CameraDialog.CameraDialogParent, IFrameCallback  {
    private static final boolean DEBUG = true;	// TODO set false when production
    private static final String TAG = "CameraDemo";

    private final Object mSync = new Object();
    // for accessing USB and USB camera
    private USBMonitor mUSBMonitor;
    private UVCCamera mUVCCamera;
    private SurfaceView mUVCCameraView;
    // for open&start / stop&close camera preview
    private ImageButton mCameraButton;
    private Surface mPreviewSurface, mconvert_surface_view;
    private boolean isActive, isPreview;

    public static final int SET_PREVIEW_WIDTH = 800;
    public static final int SET_PREVIEW_HEIGHT = 600;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "onCreate");
        setContentView(R.layout.camerademo);
        mCameraButton = (ImageButton)findViewById(R.id.camera_button);
        mCameraButton.setOnClickListener(mOnClickListener);

        mUVCCameraView = (SurfaceView)findViewById(R.id.camera_surface_view);
        mUVCCameraView.getHolder().addCallback(mSurfaceViewCallback);

        mUSBMonitor = new USBMonitor(this, mOnDeviceConnectListener);
    }

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
    private Bitmap tempBitmap;
    private ImageView uvc_image;

    private final IFrameCallback mIFrameCallback1 = new IFrameCallback() {
        @Override
        public void onFrame(final ByteBuffer frame) {
            Log.v(TAG, "onFrame Trigger.");
            synchronized (mSync) {
                final int n = frame.limit();
                if (buf_sz < n) {
                    buf_sz = n;
                    buf = new byte[n];
                }
                frame.get(buf, 0, n);

                if (tempBitmap == null) {
                    tempBitmap = Bitmap.createBitmap(UVCCamera.DEFAULT_PREVIEW_WIDTH, UVCCamera.DEFAULT_PREVIEW_HEIGHT, Bitmap.Config.RGB_565);
                }
                frame.clear();
                tempBitmap.copyPixelsFromBuffer(frame);
            }
            uvc_image.post(mUpdateImageTask);
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
                CameraDialog.showDialog(CameraDemoCopy.this);
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
            Toast.makeText(CameraDemoCopy.this, "USB_DEVICE_ATTACHED", Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onDetach(final UsbDevice device) {
            if (DEBUG) Log.v(TAG, "onDettach:");
            Toast.makeText(CameraDemoCopy.this, "USB_DEVICE_DETACHED", Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onConnect(final UsbDevice device, final USBMonitor.UsbControlBlock ctrlBlock, final boolean createNew) {
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
                        try {
                            camera.setPreviewSize(UVCCamera.DEFAULT_PREVIEW_WIDTH, UVCCamera.DEFAULT_PREVIEW_HEIGHT, UVCCamera.FRAME_FORMAT_MJPEG);
                        } catch (final IllegalArgumentException e) {
                            try {
                                // fallback to YUV mode
                                camera.setPreviewSize(UVCCamera.DEFAULT_PREVIEW_WIDTH, UVCCamera.DEFAULT_PREVIEW_HEIGHT, UVCCamera.DEFAULT_PREVIEW_MODE);
                            } catch (final IllegalArgumentException e1) {
                                camera.destroy();
                                return;
                            }
                        }
                        mPreviewSurface = mUVCCameraView.getHolder().getSurface();
                        if (mPreviewSurface != null) {
                            isActive = true;
                            camera.setPreviewDisplay(mPreviewSurface);
                            camera.startPreview();
                            isPreview = true;
                        }
                        synchronized (mSync) {
                            mUVCCamera = camera; //frame size 640*480
                            mUVCCamera.setFrameCallback(mIFrameCallback1, UVCCamera.PIXEL_FORMAT_YUV420SP);

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
}
