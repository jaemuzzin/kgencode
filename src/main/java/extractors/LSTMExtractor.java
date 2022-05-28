
package extractors;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;
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
        //todo: change start to indarray with shape [miniBatchSize(numNodes), vectorSize (1), timeSeriesLength(start dimensions)]
        int numUnits = 10;
         SDVariable cLast = start.getSameDiff().var("cLast", Nd4j.zeros(DataType.FLOAT, 1, 10));
            SDVariable yLast = start.getSameDiff().var("yLast", Nd4j.zeros(DataType.FLOAT, 1, 10));

            LSTMLayerConfig c = LSTMLayerConfig.builder()
                    .lstmdataformat(LSTMDataFormat.NTS)
                    .directionMode(LSTMDirectionMode.FWD)
                    .gateAct(LSTMActivations.SIGMOID)
                    .cellAct(LSTMActivations.TANH)
                    .outAct(LSTMActivations.TANH)
                    .retFullSequence(true)
                    .retLastC(true)
                    .retLastH(true)
                    .build();

            LSTMLayerOutputs outputs = new LSTMLayerOutputs(start.getSameDiff().rnn.lstmLayer(
                    start, cLast, yLast, null,
                    LSTMLayerWeights.builder()
                            .weights(start.getSameDiff().var("weights", Nd4j.rand(DataType.FLOAT, 1, 4 * numUnits)))
                            .rWeights(start.getSameDiff().var("rWeights", Nd4j.rand(DataType.FLOAT, numUnits, 4 * numUnits)))
                            .peepholeWeights(start.getSameDiff().var("inputPeepholeWeights", Nd4j.rand(DataType.FLOAT, 3 * numUnits)))
                            .bias(start.getSameDiff().var("bias", Nd4j.rand(DataType.FLOAT, 4 * numUnits)))
                            .build(),
                    c), c);


//           Behaviour with default settings: 3d (time series) input with shape
//          [miniBatchSize, vectorSize, timeSeriesLength] -> 2d output [miniBatchSize, vectorSize]
            SDVariable layer0 = outputs.getOutput();

            SDVariable layer1 = layer0.mean(1);

            SDVariable w1 = start.getSameDiff().var("w1", Nd4j.rand(DataType.FLOAT, numUnits, finalDimension));
            SDVariable b1 = start.getSameDiff().var("b1", Nd4j.rand(DataType.FLOAT, finalDimension));


            SDVariable out = start.getSameDiff().nn.softmax("out", layer1.mmul(w1).add(b1));
            
            //todo change out[miniBatchSize(numNodes), vectorSize(finalDimension)] to indarray wiht shape [finalDimension x numNodes]
            return out;
    }

    @Override
    public int getDimensions() {
        return finalDimension;
    }
    
}
