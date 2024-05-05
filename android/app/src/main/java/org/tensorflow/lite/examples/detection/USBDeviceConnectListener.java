package org.tensorflow.lite.examples.detection;

import android.hardware.usb.UsbDevice;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.widget.Toast;

import com.serenegiant.common.BaseActivity;
import com.serenegiant.usb.IFrameCallback;
import com.serenegiant.usb.USBMonitor;
import com.serenegiant.usb.UVCCamera;

public class USBDeviceConnectListener extends BaseActivity implements USBMonitor.OnDeviceConnectListener {

    private static final String TAG = "USBDeviceConnectListene";

    private final Object mSync = new Object();
    private UVCCamera mUVCCamera;
    private boolean isActive;
    private boolean isPreview;
    private Surface mPreviewSurface;
    private final SurfaceView mUVCCameraView;

    public USBDeviceConnectListener(Surface mPreviewSurface,
                                    SurfaceView mUVCCameraView,
                                    UVCCamera mUVCCamera, boolean isActive, boolean isPreview) {
        this.mUVCCamera = mUVCCamera;
        this.isActive = isActive;
        this.isPreview = isPreview;
        this.mPreviewSurface = mPreviewSurface;
        this.mUVCCameraView = mUVCCameraView;
    }

    private int buf_sz = 4096;
    private byte[] buf = new byte[buf_sz];

    private final IFrameCallback mIFrameCallback1 = frame -> {
        Log.v(TAG, "onFrame Trigger.");
        synchronized (mSync) {
            final int n = frame.limit();
            if (buf_sz < n) {
                buf_sz = n;
                buf = new byte[n];
            }
            frame.get(buf, 0, n);
            //Save buf on my ConcurrentLinkedDeque
        }
    };

//    @Override
//    public void onFrame(ByteBuffer byteBuffer) {
//        Log.v(TAG, "onFrame:");
//        Toast.makeText(USBDeviceConnectListener.this, "onFrame", Toast.LENGTH_SHORT).show();
//    }

    @Override
    public void onAttach(final UsbDevice device) {
        Log.v(TAG, "onAttach:");
        Toast.makeText(USBDeviceConnectListener.this, "USB_DEVICE_ATTACHED", Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onDetach(final UsbDevice device) {
        Log.v(TAG, "onDettach:");
        Toast.makeText(USBDeviceConnectListener.this, "USB_DEVICE_DETACHED", Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onConnect(final UsbDevice device, final USBMonitor.UsbControlBlock ctrlBlock, final boolean createNew) {
        Log.v(TAG, "onConnect:");
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
                    Log.i(TAG, "supportedSize:" + camera.getSupportedSize());
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
                        mUVCCamera = camera;
                        mUVCCamera.setFrameCallback(mIFrameCallback1, UVCCamera.PIXEL_FORMAT_YUV420SP);
                    }
                }
            }
        }, 0);
    }

    @Override
    public void onDisconnect(final UsbDevice device, final USBMonitor.UsbControlBlock ctrlBlock) {
        Log.v(TAG, "onDisconnect:");
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
}
