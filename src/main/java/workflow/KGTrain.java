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

    private KnowledgeGraph train;
    private KnowledgeGraph test;
    private int epochs;
    private int hops;
    private RelGNN model;
    private int maxSubgraphNodes;
    private NodeInitializer nodeInitializer;

    public KGTrain(KnowledgeGraph train, KnowledgeGraph test, int epochs, int hops, RelGNN model, int maxSubgraphNodes, NodeInitializer nodeInitializer) {
        this.train = train;
        this.test = test;
        this.epochs = epochs;
        this.hops = hops;
        this.model = model;
        this.maxSubgraphNodes = maxSubgraphNodes;
        this.nodeInitializer = nodeInitializer;
    }

    long iter = 0;
    int score = 0;

    public void trainPositivesAndNegatives() throws FileNotFoundException {
        Random r = new Random();
        for (int epoch = 0; epoch < epochs; epoch++) {
            train.getGraph().edgeSet().stream().parallel().forEach(e -> {
                MultiGraph subgraph = train.subgraph(e.h, e.t, hops, maxSubgraphNodes).toSequentialIdGraph();
                if (subgraph.getRelationCount() < 1) {
                    return;
                }
                //need to expand dims to make it a "minibatch"?
                INDArray X = nodeInitializer.embed(subgraph, train.getRelationCount(), maxSubgraphNodes);
                model.fit(X, Nd4j.createFromArray(new float[][]{{1.0f, 0.0f}}),
                        subgraph
                                .getMultiRelAdjacencyTensor(maxSubgraphNodes, train.getRelations().size()), e);

                int dummyR = r.nextInt(train.getRelationCount());
                while (dummyR == e.r) {
                    dummyR = r.nextInt(train.getRelationCount());
                }
                model.fit(X, Nd4j.createFromArray(new float[][]{{0.0f, 1.0f}}),
                        subgraph
                                .getMultiRelAdjacencyTensor(maxSubgraphNodes, train.getRelations().size()), new Triple(e.h, dummyR, e.t));

                if (iter % 100 == 0) {
                    test.getGraph().edgeSet().stream().parallel().forEach(te -> {
                        MultiGraph tsubgraph = test.subgraph(te.h, te.t, hops, maxSubgraphNodes).toSequentialIdGraph();
                        if (tsubgraph.getRelationCount() < 1) {
                            return;
                        }
                        //need to expand dims to make it a "minibatch"?
                        INDArray tX = nodeInitializer.embed(tsubgraph, train.getRelationCount(), maxSubgraphNodes);
                        int tdummyR = r.nextInt(train.getRelationCount());
                        while (tdummyR == te.r) {
                            tdummyR = r.nextInt(train.getRelationCount());
                        }
                        INDArray o = model.output(tX, tsubgraph
                                .getMultiRelAdjacencyTensor(maxSubgraphNodes, train.getRelations().size()), te);

                        INDArray n = model.output(tX, tsubgraph.getMultiRelAdjacencyTensor(maxSubgraphNodes, train.getRelations().size()), new Triple(te.h, tdummyR, te.t));
                        iter++;
                        if (o.getDouble(0, 0) > o.getDouble(0, 1)) {
                            score++;
                        }
                        if (n.getDouble(0, 0) < n.getDouble(0, 1)) {
                            score++;
                        }

                    });
                    System.out.println(score + " /" + (test.getGraph().edgeSet().size()*2));
                    score=0;
                }
            });
        }
    }
}
