package cloud.asaru.thekg;

import java.io.InputStream;
import java.util.List;
import java.util.stream.Collectors;
import org.jgrapht.Graph;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 *
 * @author Jae
 */
public class TrainerTest {
    
    public TrainerTest() {
    }
    
    @Test
    public void testTrainEncoder() {
        System.out.println("trainEncoder");
        int hops = 0;
        int epochs = 0;
        Trainer instance = new Trainer();
        instance.load(getClass().getResourceAsStream("nell.txt"));
        /*GraphEncoder encoder = new GraphEncoder(instance.getKg().entities.size(), instance.getKg().relations.size(), 30, 100);
        instance.trainEncoder(encoder, 2, 10);
        instance.getKg().graph.edgeSet().forEach(triple -> {
            Graph<Integer, Triple> g = instance.getKg().subgraph(triple.h, triple.t, hops);
            List<Triple> subEdges = g.edgeSet().stream().collect(Collectors.toList());
            subEdges.addAll(subEdges);
            Triple[] r = encoder.autoencode(subEdges.toArray(new Triple[0]), instance.getKg().entities.size(), instance.getKg().relations.size());
            assertArrayEquals(subEdges.toArray(new Triple[0]), r);
        });*/
    }
    
}
