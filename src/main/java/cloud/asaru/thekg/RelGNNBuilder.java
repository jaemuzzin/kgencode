
package cloud.asaru.thekg;

import org.nd4j.linalg.api.ndarray.INDArray;
import extractors.FeatureExtractor;
import org.nd4j.autodiff.samediff.SDVariable;

/**
 *
 * @author Jae
 */
public abstract class RelGNNBuilder {

    public RelGNNBuilder() {
    }
    public abstract RelGNN build(SDVariable input, int numRels,int numNodes, int dims, int layers, boolean learnable, boolean sigmoid, FeatureExtractor featureExtractor);
    public abstract RelGNN build(int numRels,int numNodes, int dims, int layers, boolean learnable, boolean sigmoid, FeatureExtractor featureExtractor);
}
