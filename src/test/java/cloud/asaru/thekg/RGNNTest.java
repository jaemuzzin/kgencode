/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cloud.asaru.thekg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class RGNNTest {
    
    public RGNNTest() {
    }

    @Test
    public void test1() {
        System.out.println("test1");
        INDArray input = Nd4j.createFromArray(new int[][]{{1,2,3}});
        RGNN instance = new RGNN(
                Nd4j.createFromArray(new int[][][] {{{-1,0,0},{0,-1,0},{0,0,-1}}}),//{{{0,0,0},{0,0,0},{0,0,0}},{{-1,0,0},{0,-1,0},{0,0,-1}},{{0,0,0},{0,0,0},{0,0,0}}}),
        3,1,false, false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{-1,-2,-3}});
        INDArray result = instance.features(input);
        System.out.println(result.toString());
        assertEquals(expResult.toString(), result.toString());
    }
     @Test
    public void test2() {
        System.out.println("test2");
        INDArray input = Nd4j.createFromArray(new int[][]{{1,2,3}});
        RGNN instance = new RGNN(
                Nd4j.createFromArray(new int[][][] {{{1,0,0},{0,1,0},{0,0,1}},{{1,2,0},{0,1,0},{0,0,-2}},{{1,0,0},{0,1,0},{0,0,1}}}),
        3,1,false, false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{1,4,-6}});
        INDArray result = instance.features(input);
        System.out.println(result.toString());
        assertEquals(expResult.toString(), result.toString());
    }
    
     @Test
    public void testTwoLayer() {
        System.out.println("testTwoLayer");
        INDArray input = Nd4j.createFromArray(new int[][]{{1,2,3}});
        RGNN instance = new RGNN(
                Nd4j.createFromArray(new int[][][] {{{1,0,0},{0,1,0},{0,0,1}},{{-2,0,0},{0,-2,0},{0,0,-2}},{{1,0,0},{0,1,0},{0,0,1}}}),
        3,2,false, false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{4,8,12}});
        INDArray result = instance.features(input);
        System.out.println(result.toString());
        assertEquals(expResult.toString(), result.toString());
    }
    
     @Test
    public void testTwoLayerLearnable() {
        System.out.println("testTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{1,2,3}});
        RGNN instance = new RGNN(
                Nd4j.createFromArray(new int[][][] {{{1,1,1},{1,1,1},{1,1,1}}}),
        3,2,true, false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{5,9,13}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.features(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*1000));
    }
    
    @Test
    public void testRelTwoLayerLearnable() {
        System.out.println("testMultiRelTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{1,2,3}});
        RGNN instance = new RGNN(
                Nd4j.createFromArray(new int[][][] {{{1,1,1},{1,1,1},{1,1,1}},{{1,1,1},{1,1,1},{1,1,1}}}),
        3,1,true, false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{5,9,13}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.features(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*1000));
    }
    
    
    @Test
    public void testtwoRelTwoLayerLearnableOnemissing() {
        System.out.println("testMultiRelTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{1,2,3}});
        RGNN instance = new RGNN(
                Nd4j.createFromArray(new int[][][] {{{1,1,0},{1,1,1},{0,1,1}},{{1,1,1},{1,1,1},{1,1,1}}}),
        3,1,true, false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{5,9,13}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.features(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*1000));
    }
    @Test
    public void testMultiRelTwoLayerLearnable() {
        System.out.println("testMultiRelTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{1,2,3}});
        RGNN instance = new RGNN(
                Nd4j.createFromArray(new int[][][] {{{1,0,1},{0,1,1},{1,1,1}},{{1,1,1},{1,1,0},{1,0,1}},{{1,1,0},{1,1,1},{0,1,1}},{{1,1,1},{1,1,1},{1,1,1}}}),
        3,1,true, false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{5,9,13}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.features(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*1000));
    }
    
    @Test
    public void testMultiRelTwoLayerLearnableProbability() {
        System.out.println("testMultiRelTwoLayerLearnable");
        INDArray input = Nd4j.createFromArray(new double[][]{{55,12,44}});
        RGNN instance = new RGNN(
                Nd4j.createFromArray(new int[][][] {{{1,0,1},{0,1,1},{1,1,1}},{{1,1,1},{1,1,0},{1,0,1}},{{1,1,0},{1,1,1},{0,1,1}},{{1,1,1},{1,1,1},{1,1,1}}}),
        3,1,true, false);
        INDArray expResult = Nd4j.createFromArray(new double[][]{{.5,.001,.999}});
        for(int i=0;i<3000;i++)instance.fit(input, expResult);
        INDArray result = instance.features(input);
        System.out.println(result.toString());
        assertEquals(0.0d, Math.round(expResult.sub(result).squaredDistance(Nd4j.zeros(expResult.shape()))*10));
    }
    
}
