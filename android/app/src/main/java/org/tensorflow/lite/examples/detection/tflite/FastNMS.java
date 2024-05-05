package org.tensorflow.lite.examples.detection.tflite;

import java.util.ArrayList;
import java.util.List;
public class FastNMS {
    // Define a class to represent a bounding box
    public static class Box {
        double x1, y1, x2, y2, score;
        int classId;
        String title;
        public Box(double x1, double y1, double x2, double y2, double score, int classId, String title) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
            this.classId = classId;
            this.title = title;
        }
        // Compute area of the box
        double area() {
            return (x2 - x1) * (y2 - y1);
        }
    }
    // Method to compute IoU between two boxes
    public static double computeIoU(Box box1, Box box2) {
        double x1 = Math.max(box1.x1, box2.x1);
        double y1 = Math.max(box1.y1, box2.y1);
        double x2 = Math.min(box1.x2, box2.x2);
        double y2 = Math.min(box1.y2, box2.y2);
        double interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        double unionArea = box1.area() + box2.area() - interArea;
        return interArea / unionArea;
    }
    // Method to perform Fast NMS
    public static List<Box> fastNMS(List<Box> boxes, double scoreThr, double iouThr, int topK, int maxNum) {
        boxes.sort((a, b) -> Double.compare(b.score, a.score));
        List<Box> result = new ArrayList<>();
        boolean[] suppressed = new boolean[boxes.size()];
        for (int i = 0; i < boxes.size(); i++) {
            if (suppressed[i] || boxes.get(i).score < scoreThr)
                continue;
            Box box1 = boxes.get(i);
            result.add(box1);
            if (result.size() == maxNum)
                break;
            for (int j = i + 1; j < boxes.size() && j < topK; j++) {
                Box box2 = boxes.get(j);
                if (computeIoU(box1, box2) > iouThr) {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }
}
