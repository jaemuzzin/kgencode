
package initializers;

import cloud.asaru.thekg.MultiGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class SpectralInitializer implements NodeInitializer{
    
    @Override
    public INDArray extract(MultiGraph graph, int dims, int nodes) {
        INDArray spectral = graph.getSpectralNodeCoordsSimpleGraph(dims, nodes);
        //INDArray blanks = Nd4j.zeros(dims, nodes - spectral.shape()[1]);
        return spectral;
    }
    
}
