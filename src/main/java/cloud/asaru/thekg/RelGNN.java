
package cloud.asaru.thekg;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Jae
 */
public interface RelGNN {

    /*
     * input is shape [feature index, node index], output is [2,1]
     */
    INDArray output(INDArray input, INDArray relationShipAdjTensor, Triple target);

    /*
     * input is shape [feature index, node index], output is [2,1]
     */
    void fit(INDArray input, INDArray output, INDArray relationShipAdjTensor, Triple target);

    
}
