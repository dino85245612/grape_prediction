package org.tensorflow.lite.examples.detection;

import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.SpeechRecognizer;
import android.util.Log;

import java.util.ArrayList;

public class SpeechRecognitionListener implements RecognitionListener {
    private OnCommandReceivedListener listener;

    @Override
    public void onReadyForSpeech(Bundle params) {
        Log.i("SpeechRecognitionListener", "voices: onReadyForSpeech.");

    }

    @Override
    public void onBeginningOfSpeech() {
        Log.i("SpeechRecognitionListener", "voices: User has started speaking.");

    }

    @Override
    public void onRmsChanged(float rmsdB) {
        Log.i("SpeechRecognitionListener", "voices: onRmsChanged.");
    }

    @Override
    public void onBufferReceived(byte[] buffer) {
        Log.i("SpeechRecognitionListener", "voices: onBufferReceived.");
    }

    @Override
    public void onEndOfSpeech() {
        Log.i("SpeechRecognitionListener", "voices: User has stopped speaking.");
    }

    @Override
    public void onError(int error) {
        String message;
        switch (error) {
            case SpeechRecognizer.ERROR_AUDIO:
                message = "onError voices: Audio recording error";
                break;
            case SpeechRecognizer.ERROR_CLIENT:
                message = "onError voices: Client side error";
                break;
            case SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS:
                message = "onError voices: Insufficient permissions";
                break;
            case SpeechRecognizer.ERROR_NETWORK:
                message = "onError voices: Network error";
                break;
            case SpeechRecognizer.ERROR_NETWORK_TIMEOUT:
                message = "onError voices: Network timeout";
                break;
            case SpeechRecognizer.ERROR_NO_MATCH:
                message = "onError voices: No match";
                break;
            case SpeechRecognizer.ERROR_RECOGNIZER_BUSY:
                message = "onError voices: RecognitionService busy";
                break;
            case SpeechRecognizer.ERROR_SERVER:
                message = "onError voices: Error from server";
                break;
            case SpeechRecognizer.ERROR_SPEECH_TIMEOUT:
                message = "onError voices: No speech input";
                break;
            default:
                message = "onError voices: Didn't understand, please try again.";
                break;
        }
        Log.d("SpeechRecognition", message);
    }

    @Override
    public void onResults(Bundle results) {
        ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        if (matches != null) {
            for (String result : matches) {
                if (result.equalsIgnoreCase("QR")) {
                    if (listener != null) {
                        listener.onQRCodeCommand();
                    }
                    break;
                }
            }
        }
    }

    @Override
    public void onPartialResults(Bundle partialResults) {

    }

    @Override
    public void onEvent(int eventType, Bundle params) {

    }

    public interface OnCommandReceivedListener {
        void onQRCodeCommand();
    }

    public void setOnCommandReceivedListener(OnCommandReceivedListener listener) {
        this.listener = (OnCommandReceivedListener) listener;
    }

}

