
package cloud.asaru.thekg;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Jae
 */
public interface RelGNN {

    INDArray getMultiRelAdjacencyTensor();
    void setMultiRelAdjacencyTensor(INDArray t);
    /*
     * input is shape [feature index, node index], output is same
     */
    INDArray output(INDArray input);

    /*
     * input is shape [feature index, node index]
     */
    void fit(INDArray input, INDArray output);

    
}
