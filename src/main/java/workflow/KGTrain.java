package workflow;

import initializers.NodeInitializer;
import cloud.asaru.thekg.KnowledgeGraph;
import cloud.asaru.thekg.MultiGraph;
import cloud.asaru.thekg.RelGNN;
import cloud.asaru.thekg.Triple;
import java.io.FileNotFoundException;
import java.util.Random;
import org.jgrapht.Graphs;
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
    private int embeddingDimensions;

    public KGTrain(KnowledgeGraph train, KnowledgeGraph test, int epochs, int hops, RelGNN model, int maxSubgraphNodes, NodeInitializer nodeInitializer, int embeddingDimensions) {
        this.train = train;
        this.test = test;
        this.epochs = epochs;
        this.hops = hops;
        this.model = model;
        this.maxSubgraphNodes = maxSubgraphNodes;
        this.nodeInitializer = nodeInitializer;
        this.embeddingDimensions = embeddingDimensions;
    }

    int iter = 0;

    public void trainPositivesAndNegatives() throws FileNotFoundException {
        Random r = new Random();
        for (int epoch = 0; epoch < epochs; epoch++) {
            train.getGraph().edgeSet().stream().forEach(e -> {
                //skip examples where this is the only edge for head or tail
                if(Graphs.neighborSetOf(train.getGraph(), e.h).size()==1 || Graphs.neighborSetOf(train.getGraph(), e.t).size()==1) return;
                MultiGraph subgraphUnnorm = train.subgraph(e.h, e.t, hops, maxSubgraphNodes, e.r);
                if(subgraphUnnorm.getSequentialIds(e.h, e.t).indexOf(e.h)==-1 || subgraphUnnorm.getSequentialIds(e.h, e.t).indexOf(e.t)==-1) throw new RuntimeException("Decrease hops or increase subgraph max size.");
                MultiGraph subgraph = subgraphUnnorm.toSequentialIdGraph(e.h, e.t);
                if (subgraph.getRelationCount() < 1) {
                    return;
                }
                //need to expand dims to make it a "minibatch"?
                INDArray X = nodeInitializer.embed(subgraph, embeddingDimensions, maxSubgraphNodes,
                        subgraphUnnorm.getSequentialIds(e.h, e.t).indexOf(e.h), subgraphUnnorm.getSequentialIds(e.h, e.t).indexOf(e.t));
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
                iter++;
                if (iter % 100 ==0) {

                    int score = 0;
                    int tested=0;
                    for (Triple te : test.getGraph().edgeSet()) {
                        //skip examples where this is the only edge for head or tail
                        if(Graphs.neighborSetOf(test.getGraph(), te.h).size()==1 || Graphs.neighborSetOf(test.getGraph(), te.t).size()==1) continue;
                        MultiGraph tsubgraphun = test.subgraph(te.h, te.t, hops, maxSubgraphNodes, te.r);
                        if(tsubgraphun.getSequentialIds(te.h, te.t).indexOf(te.h)==-1 || tsubgraphun.getSequentialIds(te.h, te.t).indexOf(te.t)==-1) throw new RuntimeException("Decrease hops or increase subgraph max size.");
                        MultiGraph tsubgraph = tsubgraphun.toSequentialIdGraph(te.h, te.t);
                        if (tsubgraph.getRelationCount() < 1) {
                            return;
                        }
                       
                        INDArray tX = nodeInitializer.embed(tsubgraph, embeddingDimensions, maxSubgraphNodes,
                                tsubgraphun.getSequentialIds(te.h, te.t).indexOf(te.h), tsubgraphun.getSequentialIds(te.h, te.t).indexOf(te.t));
                        int tdummyR = r.nextInt(train.getRelationCount());
                        while (tdummyR == te.r) {
                            tdummyR = r.nextInt(train.getRelationCount());
                        }
                        INDArray o = model.output(tX, tsubgraph
                                .getMultiRelAdjacencyTensor(maxSubgraphNodes, train.getRelations().size()), te);

                        INDArray n = model.output(tX, tsubgraph.getMultiRelAdjacencyTensor(maxSubgraphNodes, train.getRelations().size()), new Triple(te.h, tdummyR, te.t));
                        if (o.getDouble(0, 0) > o.getDouble(0, 1)) {
                            score++;
                        }
                        if (n.getDouble(0, 0) < n.getDouble(0, 1)) {
                            score++;
                        }
                        tested++;

                    }
                    System.out.println(score + " /" + (tested * 2));
                    iter = 0;
                }
            });
        }
    }
}
