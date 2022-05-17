package workflow;

import cloud.asaru.thekg.KnowledgeGraph;
import cloud.asaru.thekg.MultiGraph;
import cloud.asaru.thekg.RelGNN;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;

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
    public KGTrain(String testFilePath, int epochs, int hops, RelGNN model, int maxSubgraphNodes) {
        this.testFilePath = testFilePath;
        this.epochs = epochs;
        this.hops = hops;
        this.model= model;
        this.maxSubgraphNodes = maxSubgraphNodes;
    }

    public void run() throws FileNotFoundException {
        KnowledgeGraph complete = new KnowledgeGraph();
        complete.build(new InputStreamReader(new FileInputStream(testFilePath)));
        for (int epoch = 0; epoch < epochs; epoch++) {
            complete.getGraph().edgeSet().forEach(e -> {
                MultiGraph subgraph = complete.subgraph(e.h, e.t, hops, maxSubgraphNodes);
                model.setMultiRelAdjacencyTensor(subgraph
                        .toSequentialIdGraph()
                        .getMultiRelAdjacencyTensor(maxSubgraphNodes, complete.getRelations().size()));
                model.fit(input, output);
            });
        }
    }
}
