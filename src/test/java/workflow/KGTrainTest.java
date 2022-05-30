
package workflow;

import cloud.asaru.thekg.KnowledgeGraph;
import cloud.asaru.thekg.RGNNShared;
import cloud.asaru.thekg.RelGNN;
import extractors.LSTMExtractor;
import extractors.SimpleExtractor;
import initializers.SimpleInitializer;
import initializers.SpectralInitializer;
import java.io.InputStreamReader;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 *
 * @author Jae
 */
public class KGTrainTest {
    
    public KGTrainTest() {
    }

    /*@Test
    public void testTrainPositives() throws Exception {
        System.out.println("trainPositives");
        KnowledgeGraph kg = new KnowledgeGraph();
        kg.build(new InputStreamReader(getClass().getResourceAsStream("/nell.txt")));
        RelGNN gnn = new RGNNShared().build(kg.getRelations().size(), 100, 10, 4, true, true, new LSTMExtractor(10, kg.getRelations().size(), 100));
        KGTrain instance = new KGTrain(kg, 10000, 2, gnn, 100, new SpectralInitializer());
        instance.trainPositivesAndNegatives();
    }*/
    @Test
    public void testTrainPositives() throws Exception {
        System.out.println("trainPositives");
        KnowledgeGraph kg = new KnowledgeGraph();
        kg.build(new InputStreamReader(getClass().getResourceAsStream("/nell.txt")));
        RelGNN gnn = new RGNNShared().build(kg.getRelations().size(), 20, 10, 4, true, true, new SimpleExtractor(10, kg.getRelations().size(), 20));
        KGTrain instance = new KGTrain(kg, 2, 2, gnn, 20, new SpectralInitializer());
        instance.trainPositivesAndNegatives();
    }
}
