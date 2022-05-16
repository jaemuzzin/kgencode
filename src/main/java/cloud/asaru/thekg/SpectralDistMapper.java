
package cloud.asaru.thekg;

import java.util.HashMap;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.jgrapht.graph.SimpleGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
/**
 *
 * @author Jae
 */
public class SpectralDistMapper extends NodeIndexToSequenceMapper{
    private int seqSize=0;
    int[] mappings;
    public SpectralDistMapper(int seqSize, SimpleDirectedGraph<Integer, Triple> subgraph, int numNodes, int numRels) {
    this.seqSize= seqSize;
    mappings = new int[seqSize];
    //need to improve this to do the fiedler after mapping to reduce complexity
    
    //need to transopose since spectralnode returns shape [node, feature]
    INDArray fielders = KnowledgeGraph
            .getSpectralNodeCoordsSimpleGraph(KnowledgeGraph.toSimpleGraph(subgraph, numNodes, numRels),
                    numNodes, numRels).transpose();
    HashMap<Integer, Double> scores = new HashMap<>();
    subgraph.edgeSet().forEach(e-> scores.put(e.h, 0.0));
    subgraph.edgeSet().forEach(e-> scores.put(e.t, 0.0));
    
    for(int i=0;i<mappings.length;i++){
        mappings[i] = 
    }
    }
    
    @Override
    public int getSequenceSize() {
        return seqSize;
    }

    @Override
    public void setSequenceSize(int n) {
       this.seqSize = n;
    }

    @Override
    public int mapToSeq(int vertexId) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int mapToVertextId(int seq) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
