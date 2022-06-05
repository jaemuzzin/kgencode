
package workflow;

import cloud.asaru.thekg.KnowledgeGraph;
import cloud.asaru.thekg.RGNNShared;
import cloud.asaru.thekg.RelGNN;
import extractors.LSTMExtractor;
import extractors.PassthroughExtractor;
import extractors.SimpleExtractor;
import initializers.DistanceInitializer;
import initializers.OnehotInitializer;
import initializers.RandomInitializer;
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
    
    @Test
    public void testTrainPassthrough() throws Exception {
        System.out.println("testTrainPassthrough");
        KnowledgeGraph kg = new KnowledgeGraph();
        kg.build(new InputStreamReader(getClass().getResourceAsStream("/nell.txt")));
        KnowledgeGraph testkg = new KnowledgeGraph();
        testkg.build(new InputStreamReader(getClass().getResourceAsStream("/nelltest.txt")));
        int dims = 2;
        int maxNodes = 300;
        RelGNN gnn = new RGNNShared().build(kg.getRelationCount(), maxNodes, dims, 5, true, true, new PassthroughExtractor(dims, 2, maxNodes));
        KGTrain instance = new KGTrain(kg, testkg, 4, 3, gnn, maxNodes, new DistanceInitializer(), 2);
        instance.trainPositivesAndNegatives();
    }

    @Test
    public void testTrainPositives() throws Exception {
        System.out.println("testTrainPositives");
        KnowledgeGraph kg = new KnowledgeGraph();
        kg.build(new InputStreamReader(getClass().getResourceAsStream("/nell.txt")));
        KnowledgeGraph testkg = new KnowledgeGraph();
        testkg.build(new InputStreamReader(getClass().getResourceAsStream("/nelltest.txt")));
        int dims = 2;
        int maxNodes = 300;
        RelGNN gnn = new RGNNShared().build(kg.getRelationCount(), maxNodes, dims, 5, true, true, new SimpleExtractor(dims, 2, maxNodes));
        KGTrain instance = new KGTrain(kg, testkg, 4, 3, gnn, maxNodes, new DistanceInitializer(), 2);
        instance.trainPositivesAndNegatives();
    }
    
    /*
    @Test
    public void trainPositivesSimple() throws Exception {
        System.out.println("trainPositivesSimple");
        KnowledgeGraph kg = new KnowledgeGraph();
        kg.build(new InputStreamReader(getClass().getResourceAsStream("/nell.txt")));
        KnowledgeGraph testkg = new KnowledgeGraph();
        testkg.build(new InputStreamReader(getClass().getResourceAsStream("/nelltest.txt")));
        int dims = 14;
        int maxNodes = 30;
        RelGNN gnn = new RGNNShared().build(kg.getRelations().size(), maxNodes, dims, 5, true, true, new SimpleExtractor(dims, kg.getRelations().size(), maxNodes));
        KGTrain instance = new KGTrain(kg,  testkg, 1, 3, gnn, maxNodes, new RandomInitializer(), 14);
        instance.trainPositivesAndNegatives();
    }*/
    /*@Test
    public void testTrainPositivesFull() throws Exception {
        System.out.println("testTrainPositivesFull");
        KnowledgeGraph kg = new KnowledgeGraph();
        kg.build(new InputStreamReader(getClass().getResourceAsStream("/nell.txt")));
        KnowledgeGraph testkg = new KnowledgeGraph();
        testkg.build(new InputStreamReader(getClass().getResourceAsStream("/nelltest.txt")));
        int dims = 7;
        int maxNodes = 70;
        RelGNN gnn = new RGNNShared().build(kg.getRelations().size(), maxNodes, dims, 5, true, true, new LSTMExtractor(dims, kg.getRelations().size(), maxNodes));
        KGTrain instance = new KGTrain(kg, testkg, 5, 3, gnn, maxNodes, new SpectralInitializer());
        instance.trainPositivesAndNegatives();
    }
    */
}
