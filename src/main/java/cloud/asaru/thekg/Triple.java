
package cloud.asaru.thekg;

import java.util.Arrays;

/**
 *
 * @author Jae
 */
public class Triple {
    public int h;
    public int r;
    public int t;
    public int[] asArray(){
        return new int[]{h,r,t};
    }

    public Triple(int h, int r, int t) {
        this.h = h;
        this.r = r;
        this.t = t;
    }

    public int getH() {
        return h;
    }

    public void setH(int h) {
        this.h = h;
    }

    public int getR() {
        return r;
    }

    public void setR(int r) {
        this.r = r;
    }

    public int getT() {
        return t;
    }

    public void setT(int t) {
        this.t = t;
    }

    @Override
    public String toString() {
        return "Triplet{" + "h=" + h + ", r=" + r + ", t=" + t + '}';
    }

    public int[] onehot(int maxV, int maxR){
        int[] ret = new int[maxV+maxV+maxR];
        Arrays.fill(ret, 0);
        ret[h]=1;
        ret[maxV+t]=1;
        ret[maxV+maxV+r]=1;
        return ret;
    }
    public int[] onehotHead(int maxV){
        int[] ret = new int[maxV];
        Arrays.fill(ret, 0);
        ret[h]=1;
        return ret;
    }
    
    public int[] onehotTail(int maxV){
        int[] ret = new int[maxV];
        Arrays.fill(ret, 0);
        ret[t]=1;
        return ret;
    }
    
    public int[] onehotR(int maxR){
        int[] ret = new int[maxR];
        Arrays.fill(ret, 0);
        ret[r]=1;
        return ret;
    }
    @Override
    public int hashCode() {
        int hash = 5;
        hash = 53 * hash + this.h;
        hash = 53 * hash + this.r;
        hash = 53 * hash + this.t;
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Triple other = (Triple) obj;
        if (this.h != other.h) {
            return false;
        }
        if (this.r != other.r) {
            return false;
        }
        if (this.t != other.t) {
            return false;
        }
        return true;
    }
    
}
