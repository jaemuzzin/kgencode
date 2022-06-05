
package cloud.asaru.thekg;

import java.util.Arrays;
import java.util.Random;
import me.xdrop.fuzzywuzzy.FuzzySearch;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Jae
 */
public class CharEncoderTest {
    
    public static final CharEncoder instance = new CharEncoder(5, 15, 20, 65);
    
    
    public static final String data0 = "ABBBCCCDDEEEABCDE";
    
    public static final String data1 = "EEEEEEEEEEEEEEEEE";
    public static final String data2 = "BCBCBCBCBCBCBCBCBCBC";
    public static final String data3 = "DDDEEDDDEEDDDEEEE";
    public static final String data4 = "DDEECDEEEEEDDDDBEE";
    public static final String data5 = "AAAAAAAAAAABBBBBBBB";
    static {
        
        Random rand = new Random(123);
        for(int i=0;i<50000;i++){
            StringBuilder random = new StringBuilder();
            int len = 10+rand.nextInt(9);
            for(int j=0;j<len;j++) {
                random.append((char)(65+Math.min(4,Math.max(0,(int)(rand.nextGaussian()*1.5+3)))));
                
            }
            //System.out.println(random.toString());
            instance.fit(new String(random));
            if(i%100==0) {
                String result = instance.autoencode(data0);
                System.out.println(result);
                String result2 = instance.autoencode(data1);
                System.out.println(result2);
                String result3 = instance.autoencode(data2);
                System.out.println(result3);
                String result5 = instance.autoencode(data3);
                System.out.println(result5);
                String result6 = instance.autoencode(data4);
                System.out.println(result6);
                String result7 = instance.autoencode(data5);
                System.out.println(result7);
                System.out.println("");
            }
        }
    }
    public CharEncoderTest() {
    }


    /*@Test
    public void testAutoencode() {
        GraphEncoder instance = new GraphEncoder(6, 4,50, 30, 20);
        for(int i=0;i<1;i++)
            instance.fit(data);
        Triple[] result = instance.autoencode(data);
        assertArrayEquals(data, result);
        result = instance.autoencode(data2);
        assertArrayEquals(data2, result);
    }*/

    @Test
    public void testAutoencodeSmallOther2() {
        
        String result = instance.autoencode(data1);
        assertEquals(data1, result);
    }
    @Test
    public void testAutoencodeSmallOther1() {
        String result = instance.autoencode(data3);
        assertTrue(FuzzySearch.ratio(data1, result) < FuzzySearch.ratio(data3, result));
        result = instance.autoencode(data4);
        assertTrue(FuzzySearch.ratio(data3, result) < FuzzySearch.ratio(data4, result));
    }
    
    
    @Test
    public void testEmbeddingDist() {
        INDArray result0 = instance.embedding(data3);
        INDArray result1 = instance.embedding(data4);
        INDArray result2 = instance.embedding(data5);
        INDArray result3 = instance.embedding(data0);
        assertTrue(result1.distance2(result0) < result2.distance2(result0));
        assertTrue(result1.distance2(result0) < result3.distance2(result0));
    }
}
