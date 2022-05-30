package workflow;

import initializers.NodeInitializer;
import cloud.asaru.thekg.KnowledgeGraph;
import cloud.asaru.thekg.MultiGraph;
import cloud.asaru.thekg.RelGNN;
import cloud.asaru.thekg.Triple;
import java.io.FileNotFoundException;
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

    public void trainPositivesAndNegatives() throws FileNotFoundException {
        Random r = new Random();
        for (int epoch = 0; epoch < epochs; epoch++) {
            complete.getGraph().edgeSet().forEach(e -> {
                MultiGraph subgraph = complete.subgraph(e.h, e.t, hops, maxSubgraphNodes).toSequentialIdGraph();
                if(subgraph.getRelationCount()<1) return;
                //need to expand dims to make it a "minibatch"?
                INDArray X = nodeInitializer.extract(subgraph, complete.getRelationCount(), maxSubgraphNodes);
                model.fit(X, Nd4j.createFromArray(new float[][]{{1.0f, 0.0f}}),
                        subgraph
                        .getMultiRelAdjacencyTensor(maxSubgraphNodes, complete.getRelations().size()), e);
                
                int dummyR = r.nextInt(complete.getRelationCount());
               
                model.fit(X, Nd4j.createFromArray(new float[][]{{0.0f, 1.0f}}),
                        subgraph
                        .getMultiRelAdjacencyTensor(maxSubgraphNodes, complete.getRelations().size()), new Triple(e.h, dummyR, e.t));
                
                System.out.println("Positive: " + model.output(X, subgraph
                        .getMultiRelAdjacencyTensor(maxSubgraphNodes, complete.getRelations().size()), e));
                System.out.println("Negative: " + model.output(X, subgraph
                        .getMultiRelAdjacencyTensor(maxSubgraphNodes, complete.getRelations().size()), new Triple(e.h,  dummyR, e.t)));
                
            });
        }
    }
}
