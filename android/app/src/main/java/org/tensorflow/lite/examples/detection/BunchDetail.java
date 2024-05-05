package org.tensorflow.lite.examples.detection;

public class BunchDetail {
    private String BunchName;
    private String bunchId;
    private int berryNum;

    public BunchDetail() {
        this.bunchId = bunchId;
        this.BunchName = BunchName;
        this.berryNum = berryNum;
    }

    public String getBunchId() {
        return bunchId;
    }

    public void setBunchId(String bunchId) {
        this.bunchId = bunchId;
    }

    public String getBunchName() {
        return BunchName;
    }

    public void setBunchName(String BunchName) {
        this.BunchName = BunchName;
    }

    public int getberryNumN() {
        return berryNum;
    }

    public void setberryNumN(int berryNum) {
        this.berryNum = berryNum;
    }
}
