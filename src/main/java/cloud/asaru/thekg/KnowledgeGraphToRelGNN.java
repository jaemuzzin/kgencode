
package cloud.asaru.thekg;

import org.jgrapht.graph.SimpleDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class KnowledgeGraphToRelGNN {
    public static RelGNN convert(SimpleDirectedGraph<Integer, Triple> subgraph, int numLayers, int numNodes, int numRels, RelGNNBuilder builder, boolean sigmoid, NodeIndexToSequenceMapper mapper) {
        INDArray relationShipAdjTensor = Nd4j.zeros(numRels, numNodes, numNodes);
        subgraph.edgeSet().forEach(e -> {
            relationShipAdjTensor.putScalar(new int[]{e.r, mapper.mapToSeq(e.h), mapper.mapToSeq(e.t)}, 1);
            relationShipAdjTensor.putScalar(new int[]{e.r, mapper.mapToSeq(e.t), mapper.mapToSeq(e.h)}, 1);
        });
        
        return builder.build(relationShipAdjTensor, numNodes, numLayers, true, sigmoid);
    }
}
