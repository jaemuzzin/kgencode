
package cloud.asaru.thekg;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Jae
 */
public abstract class RelGNNBuilder {

    public RelGNNBuilder() {
    }
    
    public abstract RelGNN build(INDArray relationShipAdjTensor, int numNodes, int layers, boolean learnable, boolean sigmoid);
}
