package org.tensorflow.lite.examples.detection.tracking;

import android.graphics.Bitmap;
import android.graphics.RectF;

import org.tensorflow.lite.examples.detection.tflite.Classifier;

import java.util.ArrayList;
import java.util.List;

public interface Preprocess {

    List<Classifier.Recognition> results = null;
    //main method to process features
    ArrayList<ProcessData> collect_features(List<Classifier.Recognition> results, Bitmap bitmap);

    // 1. to separate the bbox to bunch and berry
    ArrayList<Classifier.Recognition> separate_bbox_cls(List<Classifier.Recognition> results, int classID);

    //2. find one bunch
    ArrayList<Classifier.Recognition> find_target_bunch(Bitmap cropBitmap, ArrayList<Classifier.Recognition> arrayList);

    //4. remove berry out of the selected bunch
    ArrayList<Classifier.Recognition>  remove_unexpected_berry(ArrayList<Classifier.Recognition> filter_berry,
                                                               ArrayList<Classifier.Recognition> selected_bunch);

    //compute iou
    float box_iou(RectF a, RectF b);
    float box_intersection(RectF a, RectF b);
    float box_union(RectF a, RectF b);
    float overlap(float x1, float w1, float x2, float w2);

    Object toBlackBitmap(Bitmap cropBitmap);
    Bitmap drawWhiteRect(Bitmap blackBitmap, ArrayList<Classifier.Recognition> bboxes);

    float findLargest(float a, float b);
    float calc_area_bbox(RectF a);
    float calc_ratio(int a, int b);

    public class ProcessData{
        private String DetectBunch = "No"; //Default
        private static int Bunch_No; //Default
        private static int selected_berrySize;
        double[] outputData;

        public ProcessData(String DetectBunch, int Bunch_No, int selected_berrySize, double[] outputData){
            this.DetectBunch = DetectBunch;
            this.Bunch_No = Bunch_No;
            this.selected_berrySize = selected_berrySize;
            this.outputData = outputData;
        }

        public int getBunchNo() {
            return Bunch_No;
        }

        public void setBunchNo(int x) {
            this.Bunch_No = x;
        }

        public int getBerySize() {
            return selected_berrySize;
        }

        public void setBerySize(int x) {
            this.selected_berrySize = x;
        }

        public double[] getOutputData() {
            return outputData;
        }

        public void SetOutputData(double[] x) {
            this.outputData = x;
        }
    }


}
