
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
    public void testTrainPositivesFull() throws Exception {
        System.out.println("testTrainPositivesFull");
        KnowledgeGraph kg = new KnowledgeGraph();
        kg.build(new InputStreamReader(getClass().getResourceAsStream("/nell.txt")));
        int dims = 7;
        int maxNodes = 70;
        RelGNN gnn = new RGNNShared().build(kg.getRelations().size(), maxNodes, dims, 5, true, true, new LSTMExtractor(dims, kg.getRelations().size(), maxNodes));
        KGTrain instance = new KGTrain(kg, 5, 3, gnn, maxNodes, new SpectralInitializer());
        instance.trainPositivesAndNegatives();
    }
    
    @Test
    public void testTrainPositives() throws Exception {
        System.out.println("trainPositives");
        KnowledgeGraph kg = new KnowledgeGraph();
        kg.build(new InputStreamReader(getClass().getResourceAsStream("/nell.txt")));
        int dims = 7;
        int maxNodes = 100;
        RelGNN gnn = new RGNNShared().build(kg.getRelations().size(), maxNodes, dims, 5, true, true, new SimpleExtractor(dims, kg.getRelations().size(), maxNodes));
        KGTrain instance = new KGTrain(kg, 5, 3, gnn, maxNodes, new SpectralInitializer());
        instance.trainPositivesAndNegatives();
    }
    
    
    @Test
    public void trainPositives() throws Exception {
        System.out.println("trainPositives");
        KnowledgeGraph kg = new KnowledgeGraph();
        kg.build(new InputStreamReader(getClass().getResourceAsStream("/nell.txt")));
        int dims = 7;
        int maxNodes = 100;
        RelGNN gnn = new RGNNShared().build(kg.getRelations().size(), maxNodes, dims, 5, true, true, new SimpleExtractor(dims, kg.getRelations().size(), maxNodes));
        KGTrain instance = new KGTrain(kg, 5, 3, gnn, maxNodes, new SimpleInitializer());
        instance.trainPositivesAndNegatives();
    }
}
