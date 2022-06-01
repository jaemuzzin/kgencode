
package initializers;

import cloud.asaru.thekg.MultiGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class RandomInitializer implements NodeInitializer{
    
    @Override
    public INDArray embed(MultiGraph graph, int dims, int nodes, int h, int v) {
        INDArray simple = Nd4j.rand(dims, nodes);
        
        return simple;
    }
    
}
