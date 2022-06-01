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
public class SimpleExtractor implements FeatureExtractor {

    private int finalDimension;
    private int startDimension;
    private int numNodes;

    public SimpleExtractor(int finalDimension, int startDimension, int numNodes) {
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
        

        
        // FinalxStart times StartxN = FinalXN
        SDVariable w1 = start.getSameDiff().var("sw1", Nd4j.rand(DataType.FLOAT, finalDimension, startDimension));
        SDVariable b1 = start.getSameDiff().var("sb1", Nd4j.zeros(DataType.FLOAT, finalDimension, 1));
        SDVariable w2 = start.getSameDiff().var("sw2", Nd4j.rand(DataType.FLOAT, numNodes, finalDimension));
        SDVariable b2 = start.getSameDiff().var("sb2", Nd4j.zeros(DataType.FLOAT, finalDimension, 1));
        SDVariable w3 = start.getSameDiff().var("sw3", Nd4j.rand(DataType.FLOAT, numNodes, finalDimension));
        SDVariable b3 = start.getSameDiff().var("sb3", Nd4j.zeros(DataType.FLOAT, finalDimension, 1));
        //NxfinalD
        SDVariable out = start.getSameDiff().nn.sigmoid(
                w3.mmul(start.getSameDiff().nn.sigmoid(
                        w2.mmul(start.getSameDiff().nn.sigmoid(
                                w1.mmul(start).add(b1).add(b2))).add(b3))));
        return out;
        
        //return start;
    }

    @Override
    public int getDimensions() {
        return finalDimension;
    }

}
