
package cloud.asaru.thekg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class RGNNSharedTest {
    
    public RGNNSharedTest() {
    }

   
    
     @Test
    public void testTwoLayerLearnable() {
        System.out.println("testTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{1,2,3}});
        RGNNShared instance = new RGNNShared(
                Nd4j.createFromArray(new int[][][] {{{1,1,1},{1,1,1},{1,1,1}}}),
        3,2,false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{5,9,13}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.output(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*1000));
    }
    
    @Test
    public void testRelTwoLayerLearnable() {
        System.out.println("testMultiRelTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{1,2,3}});
        RGNNShared instance = new RGNNShared(
                Nd4j.createFromArray(new int[][][] {{{1,1,1},{1,1,1},{1,1,1}},{{1,1,1},{1,1,1},{1,1,1}}}),
        3,1,false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{5,9,13}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.output(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*1000));
    }
    
    
    @Test
    public void testtwoRelTwoLayerLearnableOnemissing() {
        System.out.println("testMultiRelTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{1,2,3}});
        RGNNShared instance = new RGNNShared(
                Nd4j.createFromArray(new int[][][] {{{1,1,0},{1,1,1},{0,1,1}},{{1,1,1},{1,1,1},{1,1,1}}}),
        3,1,false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{5,9,13}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.output(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*1000));
    }
    @Test
    public void testMultiRelTwoLayerLearnable() {
        System.out.println("testMultiRelTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{1,2,3}});
        RGNNShared instance = new RGNNShared(
                Nd4j.createFromArray(new int[][][] {{{1,0,1},{0,1,1},{1,1,1}},{{1,1,1},{1,1,0},{1,0,1}},{{1,1,0},{1,1,1},{0,1,1}},{{1,1,1},{1,1,1},{1,1,1}}}),
        3,1,false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{5,9,13}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.output(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*1000));
    }
    
    @Test
    public void testMultiRelTwoLayerLearnableProbability() {
        System.out.println("testMultiRelTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{55,12,44}});
        RGNNShared instance = new RGNNShared(
                Nd4j.createFromArray(new int[][][] {{{1,0,1},{0,1,1},{1,1,1}},{{1,1,1},{1,1,0},{1,0,1}},{{1,1,0},{1,1,1},{0,1,1}},{{1,1,1},{1,1,1},{1,1,1}}}),
        3,1,true);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{.5,.001,.999}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.output(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*10));
    }
    
}
