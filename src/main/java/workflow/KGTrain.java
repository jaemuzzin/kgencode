package workflow;

import extractors.FeatureExtractor;
import initializers.NodeInitializer;
import cloud.asaru.thekg.KnowledgeGraph;
import cloud.asaru.thekg.MultiGraph;
import cloud.asaru.thekg.RelGNN;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Admin
 */
public class KGTrain {

    private String testFilePath;
    private int epochs;
    private int hops;
    private RelGNN model;
    private int maxSubgraphNodes;
    private FeatureExtractor featureExtractor;
    private NodeInitializer nodeInitializer;

    public KGTrain(String testFilePath, int epochs, int hops, RelGNN model, int maxSubgraphNodes, FeatureExtractor featureExtractor, NodeInitializer nodeInitializer) {
        this.testFilePath = testFilePath;
        this.epochs = epochs;
        this.hops = hops;
        this.model = model;
        this.maxSubgraphNodes = maxSubgraphNodes;
        this.featureExtractor = featureExtractor;
        this.nodeInitializer = nodeInitializer;
    }

    public void trainPositives() throws FileNotFoundException {
        KnowledgeGraph complete = new KnowledgeGraph();
        complete.build(new InputStreamReader(new FileInputStream(testFilePath)));
        for (int epoch = 0; epoch < epochs; epoch++) {
            complete.getGraph().edgeSet().forEach(e -> {
                MultiGraph subgraph = complete.subgraph(e.h, e.t, hops, maxSubgraphNodes);
                model.setMultiRelAdjacencyTensor(subgraph
                        .toSequentialIdGraph()
                        .getMultiRelAdjacencyTensor(maxSubgraphNodes, complete.getRelations().size()));
                INDArray X = nodeInitializer.extract(subgraph);
                model.fit(X, Nd4j.createFromArray(new double[]{1, 0}));
            });
        }
    }
}
