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
public class LSTMExtractor implements FeatureExtractor {

    private int finalDimension;
    private int startDimension;
    private int numNodes;

    public LSTMExtractor(int finalDimension, int startDimension, int numNodes) {
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
        int numUnits = 7;
        //change start to indarray with shape [timeSeriesLength(start dimensions), miniBatchSize(numNodes), vectorSize (1)]
        SDVariable in = start.getSameDiff().expandDims(start, 2);
        SDVariable cLast = start.getSameDiff().var("cLast", Nd4j.zeros(DataType.FLOAT, numNodes, numUnits));
        SDVariable yLast = start.getSameDiff().var("yLast", Nd4j.zeros(DataType.FLOAT, numNodes, numUnits));

        LSTMLayerConfig c = LSTMLayerConfig.builder()
                .lstmdataformat(LSTMDataFormat.TNS)
                .directionMode(LSTMDirectionMode.FWD)
                .gateAct(LSTMActivations.SIGMOID)
                .cellAct(LSTMActivations.TANH)
                .outAct(LSTMActivations.TANH)
                .retFullSequence(true)
                .retLastC(true)
                .retLastH(true)
                .build();

        LSTMLayerOutputs outputs = new LSTMLayerOutputs(start.getSameDiff().rnn.lstmLayer(
                in, cLast, yLast, null,
                LSTMLayerWeights.builder()
                        .weights(start.getSameDiff().var("weights", Nd4j.rand(DataType.FLOAT, 1, 4 * numUnits)))
                        .rWeights(start.getSameDiff().var("rWeights", Nd4j.rand(DataType.FLOAT, numUnits, 4 * numUnits)))
                        .peepholeWeights(start.getSameDiff().var("inputPeepholeWeights", Nd4j.rand(DataType.FLOAT, 3 * numUnits)))
                        .bias(start.getSameDiff().var("bias", Nd4j.rand(DataType.FLOAT, 4 * numUnits)))
                        .build(),
                c), c);

//           Behaviour with default settings: 3d (time series) input with shape
//          [timeSeriesLength, miniBatchSize, vectorSize] -> 2d output [miniBatchSize, vectorSize]
//          [startDims, Nodes, 1] -> 2d output [Nodes, finalDimension]
        SDVariable layer0 = outputs.getOutput();
        SDVariable layer1 = layer0.mean(0);
        //Nxunits
        
        SDVariable w1 = start.getSameDiff().var("w1", Nd4j.rand(DataType.FLOAT, numUnits, finalDimension));
        SDVariable b1 = start.getSameDiff().var("b1", Nd4j.rand(DataType.FLOAT, finalDimension));
        //NxfinalD
        SDVariable out = start.getSameDiff().nn.softmax("out", start.getSameDiff().transpose(layer1.mmul(w1).add(b1)));

        return out;
    }

    @Override
    public int getDimensions() {
        return finalDimension;
    }

}
