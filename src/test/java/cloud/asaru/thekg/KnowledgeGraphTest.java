
package cloud.asaru.thekg;

import java.io.Reader;
import java.util.ArrayList;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.jgrapht.graph.SimpleGraph;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Jae
 */
public class KnowledgeGraphTest {
    
    public KnowledgeGraphTest() {
    }


    @Test
    public void testToSimpleGraph() {
        System.out.println("toSimpleGraph");
       
        SimpleGraph<Integer, DefaultEdge> exp = new SimpleGraph<>(DefaultEdge.class);
        exp.addVertex(1);
        exp.addVertex(2);
        exp.addEdge(2, 1);
        assertTrue(exp.containsEdge(1, 2));
    }
    
}
