
package initializers;

import cloud.asaru.thekg.MultiGraph;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Jae
 */
public interface NodeInitializer {
    
    //returns and DxN matrix of features for each node
    public INDArray extract(MultiGraph graph);
    
    public long getDimensions();
}