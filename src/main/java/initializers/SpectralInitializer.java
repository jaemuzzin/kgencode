
package initializers;

import cloud.asaru.thekg.MultiGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class SpectralInitializer implements NodeInitializer{
MultiGraph graph;
    @Override
    public INDArray extract(MultiGraph graph) {
        this.graph = graph;
        INDArray feats = Nd4j.zeros(getDimensions(), graph.getNodeCount());
        INDArray spectral = graph.toSequentialIdGraph().getSpectralNodeCoordsSimpleGraph((int)getDimensions());
        return spectral;
    }

    @Override
    public long getDimensions() {
        if(graph!=null){
            return graph.getRelationCount();
        }
        return 0;
    }
    
}
