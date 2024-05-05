package org.tensorflow.lite.examples.detection.tracking;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Build;

import androidx.annotation.RequiresApi;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PreprocessFeature implements  Preprocess{
    private String DetectBunch;
    private int Bunch_No;
    private static final Logger LOGGER = new Logger();

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public ArrayList<ProcessData> collect_features(List<Classifier.Recognition> results, Bitmap cropBitmap) {
        LOGGER.i("Preprocess results:" + results);
        // 1. to separate the bbox to bunch and berry
        //getDetectedClass : 0-bunch, 1-berry
        ArrayList<Classifier.Recognition> filter_bunch = separate_bbox_cls(results, 0);
        ArrayList<Classifier.Recognition> filter_berry = separate_bbox_cls(results, 1);
        //LOGGER.i("prediction bunch_bboxes:" + filter_bunch.getClass());
        //LOGGER.i("prediction berry_bboxes:" + filter_berry.size());

        //2. find one bunch
        ArrayList<Classifier.Recognition> selected_bunch = find_target_bunch(cropBitmap, filter_bunch);
        //LOGGER.i("selected_bunch:" + selected_bunch);

        //3. check is there recognise a bunch
        if (selected_bunch.size() != 0){
            DetectBunch = "Yes";
            Bunch_No = selected_bunch.size();
        }
        else{
            DetectBunch = "No";
            Bunch_No = selected_bunch.size();
        }

        //4. remove berry out of the selected bunch
        ArrayList<Classifier.Recognition>  selected_berry;
        selected_berry = (ArrayList<Classifier.Recognition>) remove_unexpected_berry(filter_berry, selected_bunch);
        //LOGGER.i("selected_berry:" + selected_berry);
        //LOGGER.i("selected_berry size:" + selected_berry.size());

        //5. images area
        int image_area = cropBitmap.getWidth() * cropBitmap.getHeight();
        //int image_area = sourceBitmap.getWidth() * sourceBitmap.getHeight();

        OpenCVLoader.initDebug();
        //6. mask selected bunch into black bitmap, and get the area of white bbox
        //a. create an black bitmap first, from cropBitmap
        Bitmap blackBitmap = (Bitmap) toBlackBitmap(cropBitmap);
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
        double bunch_ratio = (calc_ratio(bunch_areas, image_area))*100;

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
        double bunchBerry_ratio = (calc_ratio(berry_areas, bunch_areas))*100;


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
            else{
                remove_index.add(i);
            }
        }
        Collections.sort(remove_index, Collections.reverseOrder());
        //LOGGER.i("remove_index:" + remove_index.size());
        //LOGGER.i("remove_index:" + remove_index);

        //11. copy selected_berry & remove the index
        ArrayList<Classifier.Recognition> remain_berry = new ArrayList<Classifier.Recognition>();
        remain_berry = (ArrayList<Classifier.Recognition>) selected_berry.clone();
        for(int i=0; i<remove_index.size(); i++){
            int index = (int) remove_index.get(i);
            //LOGGER.i("remove index:" + index);
            Classifier.Recognition remove = remain_berry.remove(index);
        }
        //LOGGER.i("selected_berry size:" + selected_berry.size());
        //LOGGER.i("remain berry size:" + remain_berry.size());

        //12. partially_overlap,	no overlap %
        List partially_overlap_index = new ArrayList();
        List no_overlap_index = new ArrayList();
        final float[] iou = new float[1];
        int overlap_no = 0;
        for (int i = 0; i < remain_berry.size(); i++) {
            final RectF location = remain_berry.get(i).getLocation();
            for (int j=0; j <remain_berry.size(); j++) {
                if (i!=j){
                    final RectF location2 = remain_berry.get(j).getLocation();
                    iou[0] = box_iou(location,location2);
                    //LOGGER.i("iou[0]:" + iou[0]);
                    if (iou[0] != 0.0){
                        overlap_no += 1;
                    }

                }

            }
            if (overlap_no == 0){
                no_overlap_index.add(i);
            }
            else if (overlap_no >= 4){
                partially_overlap_index.add(i);
            }
        }
        //LOGGER.i("no_overlap_index:" + no_overlap_index);
        //LOGGER.i("no_overlap_index size:" + no_overlap_index.size());
        //LOGGER.i("partially_overlap_index:" + partially_overlap_index);
        //LOGGER.i("partially_overlap_index size:" + partially_overlap_index.size());

        float no_ratio = (float) 0.0;
        if (no_overlap_index.size() != 0){
            no_ratio = (float)(no_overlap_index.size()/selected_berry.size())*100;
        }
        else no_ratio = (float) 0.0;

        /*
        LOGGER.i("no_ratio:" + no_ratio);
        LOGGER.i("image_area:" + image_area);
        LOGGER.i("bunch_areas:" + bunch_areas);
        LOGGER.i("bunch_ratio:" + bunch_ratio);
        LOGGER.i("bunchBerry_ratio:" + bunchBerry_ratio);

         */

        //13. input to rf
        //Detected_Berry	bunch_ratio	bunchBerry_ratio	partially_overlap	no overlap %
        double inputData[] = new double[5];
        DecimalFormat df = new DecimalFormat("###.##");
        if (selected_bunch.size() != 0) {
            inputData[0] = selected_berry.size();//36;
            inputData[1] = Double.parseDouble(df.format(bunch_ratio));//(double) 10.48;
            inputData[2] = Double.parseDouble(df.format(bunchBerry_ratio));//(double) 49.64;
            inputData[3] = partially_overlap_index.size();//(double) 3.0;
            inputData[4] = no_ratio;//(double) 11.11;
            /*
            LOGGER.i("inputData[0]:" + inputData[0]);
            LOGGER.i("inputData[1]:" + inputData[1]);
            LOGGER.i("inputData[2]:" + inputData[2]);
            LOGGER.i("inputData[3]:" + inputData[3]);
            LOGGER.i("inputData[4]:" + inputData[4]);
            LOGGER.i("no_ratio:" + no_ratio);
            LOGGER.i("image_area:" + image_area);
            LOGGER.i("bunch_areas:" + bunch_areas);
            LOGGER.i("bunch_ratio:" + bunch_ratio);
            LOGGER.i("bunchBerry_ratio:" + bunchBerry_ratio);
             */
        }else{
            inputData[0] = 0;
            inputData[1] = (double) 0.0;
            inputData[2] = (double) 0.0;
            inputData[3] = (double) 0.0;
            inputData[4] = (double) 0.0;

        }
        int num_berry = selected_berry.size();
        ArrayList<ProcessData> computeData = new ArrayList<ProcessData>();
        computeData.add(new ProcessData(DetectBunch, Bunch_No, num_berry, inputData));
        //compute.setBerySize(num_berry);
        //compute.setBunchNo(Bunch_No);
        //compute.SetOutputData(inputData);
        //ArrayList<Preprocess> detections = new ArrayList<Preprocess>();// + num_berry + inputData[
        //detections.add(new Preprocess(Bunch_No, num_berry, inputData));
        return computeData;

    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public ArrayList<Classifier.Recognition> separate_bbox_cls(List<Classifier.Recognition> results, int classID) {
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

    @Override
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

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public ArrayList<Classifier.Recognition> remove_unexpected_berry(ArrayList<Classifier.Recognition> filter_berry, ArrayList<Classifier.Recognition> selected_bunch) {
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

    @Override
    public float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    @Override
    public float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    @Override
    public float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    @Override
    public float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    @Override
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

    @Override
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

    @Override
    public float findLargest(float a, float b) {
        if (a > b){ return a; }
        else return b;
    }

    @Override
    public float calc_area_bbox(RectF a) {
        //float area = (a.right - a.left) * (a.bottom - a.top);
        return (a.right - a.left) * (a.bottom - a.top);
    }

    @Override
    public float calc_ratio(int a, int b) {
        return (float)a/b;
    }
}
