package workflow;

import extractors.FeatureExtractor;
import initializers.NodeInitializer;
import cloud.asaru.thekg.KnowledgeGraph;
import cloud.asaru.thekg.MultiGraph;
import cloud.asaru.thekg.RelGNN;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Admin
 */
public class KGTrain {

    private KnowledgeGraph complete;
    private int epochs;
    private int hops;
    private RelGNN model;
    private int maxSubgraphNodes;
    private NodeInitializer nodeInitializer;

    public KGTrain(KnowledgeGraph complete, int epochs, int hops, RelGNN model, int maxSubgraphNodes, NodeInitializer nodeInitializer) {
        this.complete = complete;
        this.epochs = epochs;
        this.hops = hops;
        this.model = model;
        this.maxSubgraphNodes = maxSubgraphNodes;
        this.nodeInitializer = nodeInitializer;
    }

    public void trainPositives() throws FileNotFoundException {
        Random r = new Random();
        for (int epoch = 0; epoch < epochs; epoch++) {
            complete.getGraph().edgeSet().forEach(e -> {
                MultiGraph subgraph = complete.subgraph(e.h, e.t, hops, maxSubgraphNodes);
                INDArray X = nodeInitializer.extract(subgraph);
                model.fit(X, Nd4j.createFromArray(new double[]{1, 0}),
                        subgraph
                        .toSequentialIdGraph()
                        .getMultiRelAdjacencyTensor(maxSubgraphNodes, complete.getRelations().size()));
                
                MultiGraph negsubgraph = complete.subgraph(e.h, r.nextInt(complete.getEntities().size()), hops, maxSubgraphNodes);
                INDArray nX = nodeInitializer.extract(subgraph);
                model.fit(nX, Nd4j.createFromArray(new double[]{0, 1}),
                        subgraph
                        .toSequentialIdGraph()
                        .getMultiRelAdjacencyTensor(maxSubgraphNodes, complete.getRelations().size()));
            });
        }
    }
}
