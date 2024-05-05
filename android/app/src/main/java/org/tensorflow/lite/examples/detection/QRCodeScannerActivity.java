package org.tensorflow.lite.examples.detection;

import android.app.Activity;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;

import androidx.annotation.Nullable;

import com.journeyapps.barcodescanner.BarcodeCallback;
import com.journeyapps.barcodescanner.BarcodeResult;
import com.journeyapps.barcodescanner.CaptureManager;
import com.journeyapps.barcodescanner.DecoratedBarcodeView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;

public class QRCodeScannerActivity extends Activity implements BarcodeCallback {
    private CaptureManager capture;
    private DecoratedBarcodeView barcodeScannerView;
    int average;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_qr_code_scanner);
        average = getIntent().getIntExtra("average", 0);

        barcodeScannerView = findViewById(R.id.zxing_barcode_scanner);
        capture = new CaptureManager(this, barcodeScannerView);
        capture.initializeFromIntent(getIntent(), savedInstanceState);

        // Add this line to set the BarcodeCallback
        barcodeScannerView.decodeSingle(this);

    }

    @Override
    protected void onResume() {
        super.onResume();
        capture.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        capture.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        capture.onDestroy();
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        capture.onSaveInstanceState(outState);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        capture.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    public void barcodeResult(BarcodeResult result) {
        if (result != null) {
            // Get the scanned information as a string
            String scannedInfo = result.getText();
            Log.d("QRCodeScanner", "Scanned information: " + scannedInfo);

            String[] df = scannedInfo.split("/");
            // Extract the cropId from the scannedInfo
            int cropId = Integer.parseInt(df[df.length-1]);

            // Create the postData JSON object
            JSONObject postData = new JSONObject();
            try {
                postData.put("crop_id", cropId);
                postData.put("grain_count", average);
                JSONObject cultivation_data = new JSONObject();
                cultivation_data.put("cultivation_method_id", 1);
                cultivation_data.put("new_status", true);
                JSONArray cultivation_list  = new JSONArray();
                cultivation_list.put(cultivation_data);
                postData.put("cultivation_statuses", cultivation_list);
                Log.i("QRCodeScanner", "crop_id" + String.valueOf(cropId));
                Log.i("QRCodeScanner", "average" + String.valueOf(average));
            } catch (JSONException e) {
                Log.e("QRCodeScanner", "Error creating postData JSON", e);
                barcodeScannerView.decodeSingle(this);
                return;
            }

            // Use this line to use the mock server
            MockWebServer mockWebServer = getMockWebServer();
            Log.i("QRCodeScanner", "here mockWebServer");

            String url = "https://db.smgrobo.jp/api/v1/grapes/update_cultivation_status/";
            Log.i("QRCodeScanner", "url" + url);
            new SendHttpPostRequestTask().execute(url, postData.toString());
            Log.i("QRCodeScanner", "here SendHttpPostRequestTask");
        } else {
            // Handle error or try again
            barcodeScannerView.decodeSingle(this);
        }
    }

    private MockWebServer getMockWebServer() {
        MockWebServer server = new MockWebServer();
        // Set the desired response
        MockResponse response = new MockResponse()
                .setResponseCode(200)
                .setBody("OK");
        server.enqueue(response);
        return server;
    }

    private void sendHttpPostRequest(String url, JSONObject json) {
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.parse("application/json; charset=utf-8");
        RequestBody body = RequestBody.create(json.toString(), JSON);

        Request request = new Request.Builder()
                .url(url)
                .post(body)
                .build();

        Log.i("QRCodeScanner", "sendHttpPostRequest before send");

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.i("QRCodeScanner", "sendHttpPostRequest onFailure");
                e.printStackTrace();
                Log.e("QRCodeScanner", "Error sending HTTP POST request");
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
//                Log.i("QRCodeScanner", "sendHttpPostRequest onResponse: " + String.valueOf(response.code()));
                Log.i("QRCodeScanner", "sendHttpPostRequest onResponse: " + String.valueOf(response.body().toString()));
                if (response.isSuccessful()) {
                    Log.d("QRCodeScanner", "HTTP POST request successful");

                    // Pass the scanned information and the average value back to CameraActivity
                    Intent resultIntent = new Intent();
                    resultIntent.putExtra("average", average);
//                    resultIntent.putExtra("scannedInfo", scannedInfo);
                    setResult(RESULT_OK, resultIntent);

                } else {
                    Log.e("QRCodeScanner", "Error: " + response.message());
                    setResult(RESULT_CANCELED); // Set failure result
                }finish(); // End the activity
            }
        });
    }

    private class SendHttpPostRequestTask extends AsyncTask<String, Void, Void> {
        @Override
        protected Void doInBackground(String... params) {
            String url = params[0];
            String postDataString = params[1];
            try {
                Log.i("QRCodeScanner", "inside SendHttpPostRequestTask ");
                JSONObject postData = new JSONObject(postDataString);
                sendHttpPostRequest(url, postData);
            } catch (JSONException e) {
                Log.e("QRCodeScanner", "Error parsing postData JSON", e);
            }
            return null;
        }
    }


}