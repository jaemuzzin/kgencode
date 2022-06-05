
package cloud.asaru.thekg;

import extractors.FeatureExtractor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMActivations;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDirectionMode;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.LSTMLayerOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.factory.Nd4j;
import sun.security.jca.GetInstance;

/**
 *
 * @author Jae
 */
public class DeepSubgraph {
    private int subgraphSize = 100;
        int queueSize = 1000;KnowledgeGraph complete;
    /**
     *
     */RGNNShared gnn;
     int hops=3;
    public DeepSubgraph(SameDiff sd, KnowledgeGraph complete,int hops, int subgraphSize, int rnnDims, int gnnDims, int layers, boolean sigmoid, FeatureExtractor featureExtractor) {
        this.subgraphSize = subgraphSize;
        this.complete=complete;
        this.hops = hops;
        //batch size, num units
        SDVariable cLast = sd.var("cLast", Nd4j.zeros(DataType.FLOAT, 1, rnnDims));
        SDVariable yLast = sd.var("yLast", Nd4j.zeros(DataType.FLOAT, 1, rnnDims));
        
        SDVariable lstmInput = sd.constant("input", Nd4j.zeros(DataType.FLOAT, queueSize, 1, complete.getNodeCount()));
        
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

        LSTMLayerOutputs outputs = new LSTMLayerOutputs(sd.rnn.lstmLayer(
                lstmInput, cLast, yLast, null,
                LSTMLayerWeights.builder()
                        .weights(sd.var("weights", Nd4j.rand(DataType.FLOAT, complete.getNodeCount(), 4 * rnnDims)))//input, 4*hidden_size
                        .rWeights(sd.var("rWeights", Nd4j.rand(DataType.FLOAT, rnnDims, 4 * rnnDims)))
                        .peepholeWeights(sd.var("inputPeepholeWeights", Nd4j.rand(DataType.FLOAT, 3 * rnnDims)))
                        .bias(sd.var("bias", Nd4j.rand(DataType.FLOAT, 4 * rnnDims)))
                        .build(),
                c), c);

//           Behaviour with default settings: 3d (time series) input with shape
//          [q, 1, nodes] -> 2d output [dims, subgraphsize]
        SDVariable layer0 = outputs.getOutput();
        SDVariable layer1 = layer0.mean(0);
        //dimensions x subgraphsize
        SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, gnnDims, rnnDims));
        SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, 1, subgraphSize));
        //NxfinalD
        SDVariable out = sd.nn.softmax("out",w1.mmul(layer1).add(b1));
        gnn = (RGNNShared) new RGNNShared().build(out, complete.getRelationCount(), (int)complete.getNodeCount(), gnnDims, layers, true, sigmoid, featureExtractor);
        
    }

    public INDArray output(int u, int v, INDArray relationShipAdjTensor, Triple target) {
        return gnn.output(Nd4j.createFromArray(new int []{u, v}), relationShipAdjTensor, target);
    }

    public void fit(int u, int v, INDArray output, INDArray relationShipAdjTensor, Triple target) {
        //need to populate candidate subgraph as 
        //[q, dims, subgraphsize]
        int i=0;
        INDArray input = Nd4j.zeros(queueSize, 1, 5);
        while(i < queueSize){
            
            i++;
        }
        complete.getMinimumHopsBetween(u, v);
        gnn.fit(Nd4j.createFromArray(new int []{u, v}), output, relationShipAdjTensor, target);
    }
    
   
    
}
