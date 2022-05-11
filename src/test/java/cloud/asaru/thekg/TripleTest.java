/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cloud.asaru.thekg;

import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class TripleTest {

    public TripleTest() {
    }

    @Test
    public void testAsArray() {
        assertEquals("12345", IntStream.range(1, 6).mapToObj(i -> "" + ((char) (i + 48))).reduce("", (a, b) -> a + b));
    }

    @Test
    public void testSanity() {
        //conclusion, input is on the left and is a row vector
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.constant("A", Nd4j.createFromArray(new float[][]{{1, 2}, {3, 4}}));
        SDVariable in = sd.placeHolder("X", DataType.FLOAT, -1, 2);
        SDVariable r = in.mmul(a);
        sd.getVariable("X").setArray(Nd4j.createFromArray(new float[][]{{5, 6}}));
        assertEquals(23, r.eval().getFloat(0, 0));
        assertEquals(34, r.eval().getFloat(0, 1));
    }

    @Test
    public void testTensorAlong() {
        INDArray t = Nd4j.createFromArray(new int[][][]{{{10, 20, 30}, {40, 50, 60}, {70, 80, 90}}, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{100, 200, 300}, {400, 500, 600}, {700, 800, 900}}});
        assertEquals(200, t.tensorAlongDimension(2, 1, 2).getDouble(0, 1));
    }
}
