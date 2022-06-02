
package cloud.asaru.thekg;

import org.nd4j.linalg.api.ndarray.INDArray;
import extractors.FeatureExtractor;

/**
 *
 * @author Jae
 */
public abstract class RelGNNBuilder {

    public RelGNNBuilder() {
    }
    
    public abstract RelGNN build(int numRels,int startDimensions, int numNodes, int dims, int layers, boolean learnable, boolean sigmoid, FeatureExtractor featureExtractor);
}
