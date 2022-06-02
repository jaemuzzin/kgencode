package extractors;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMActivations;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDirectionMode;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.LSTMLayerOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class PassthroughExtractor implements FeatureExtractor {

    private int finalDimension;
    private int startDimension;
    private int numNodes;

    public PassthroughExtractor(int finalDimension, int startDimension, int numNodes) {
        this.finalDimension = finalDimension;
        this.startDimension = startDimension;
        this.numNodes = numNodes;
    }

    /*
    * start has shape startDimensions x numNodes
     */
    @Override
    public SDVariable extract(SDVariable start) {
        if (start.getShape()[0] != startDimension) {
            throw new IllegalArgumentException("shape must be startDimensions x numNodes");
        }
        if (start.getShape()[1] != numNodes) {
            throw new IllegalArgumentException("shape must be startDimensions x numNodes");
        }
        

        
        return start;
        
        //return start;
    }

    @Override
    public int getDimensions() {
        return finalDimension;
    }

}
