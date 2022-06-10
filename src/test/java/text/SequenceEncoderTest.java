/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package text;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Jae
 */
public class SequenceEncoderTest {
    
    public SequenceEncoderTest() {
    }

    @Test
    public void testFit() {
        System.out.println("fit");
        INDArray s = null;
        SequenceEncoder instance = null;
        instance.fit(s);
        fail("The test case is a prototype.");
    }

    @Test
    public void testAutoencode() {
        System.out.println("autoencode");
        INDArray s = null;
        SequenceEncoder instance = null;
        INDArray expResult = null;
        INDArray result = instance.autoencode(s);
        assertEquals(expResult, result);
        fail("The test case is a prototype.");
    }

    @Test
    public void testEmbedding() {
        System.out.println("embedding");
        INDArray s = null;
        SequenceEncoder instance = null;
        INDArray expResult = null;
        INDArray result = instance.embedding(s);
        assertEquals(expResult, result);
        fail("The test case is a prototype.");
    }
    
}
