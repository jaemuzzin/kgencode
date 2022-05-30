
package initializers;

import cloud.asaru.thekg.MultiGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class SimpleInitializer implements NodeInitializer{
    
    @Override
    public INDArray extract(MultiGraph graph, int dims, int nodes) {
        INDArray simple = Nd4j.rand(dims, nodes);
        
        return simple;
    }
    
}
