package cloud.asaru.thekg;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;
import extractors.FeatureExtractor;
import org.nd4j.linalg.dataset.MultiDataSet;

/**
 *
 * @author Jae
 */
public class RGNNShared extends RelGNNBuilder implements RelGNN {

    public RGNNShared() {
    }

    @Override
    public RelGNN build(SDVariable input, int numRels,  int numNodes, int dims, int layers, boolean learnable, boolean sigmoid, FeatureExtractor featureExtractor) {
        return new RGNNShared(input, numRels,  numNodes, dims, layers, sigmoid, featureExtractor);
    }
    @Override
    public RelGNN build(int numRels, int numNodes, int dims, int layers, boolean learnable, boolean sigmoid, FeatureExtractor featureExtractor) {
        return new RGNNShared(numRels, numNodes, dims, layers, sigmoid, featureExtractor);
    }

    int numNodes;
    int dims;
    SameDiff sd;
    SDVariable softmax;
    SDVariable identity;
    SDVariable sigmoid;
    SDVariable label;
    int startDimensions;
    int numRels;
    int layers;
    Triple target;
    
    /*
    * in should be shape [startDimensions, numNodes]
    */
    protected RGNNShared(SDVariable in, int numRels,  int numNodes, int dims, int layers, boolean sigmoid, FeatureExtractor featureExtractor) {
        init(in, numRels, numNodes, dims, layers, sigmoid, featureExtractor);
    }
    protected RGNNShared(int numRels, int numNodes, int dims, int layers, boolean sigmoid, FeatureExtractor featureExtractor) {
        init(sd.placeHolder("input", DataType.FLOAT, numNodes), numRels, numNodes, dims, layers, sigmoid, featureExtractor);
    }
    private void init(SDVariable in, int numRels, int numNodes, int dims, int layers, boolean sigmoid, FeatureExtractor featureExtractor) {
        this.numRels = numRels;
        this.numNodes = numNodes;
        this.layers = layers;
        this.dims = dims;
        sd = SameDiff.create();
        //the shared wieghts
        //one-hot vector of relationship identity
        SDVariable rt = sd.constant("rt", Nd4j.zeros(dims, numRels));
        SDVariable X = featureExtractor.extract(in);
        SDVariable beta = sd.var("beta", new XavierInitScheme('c', dims, dims), DataType.FLOAT, dims, dims);
        label = sd.placeHolder("label", DataType.FLOAT,1, 2);
        SDVariable last = X;
        for (int layer = 0; layer < layers; layer++) {
            for (int r = 0; r < numRels; r++) {
                /* the trained per-relationship linear combination of beta */
                SDVariable alpha = sd.var("alpha_" + r + "_" + layer, Nd4j.ones(dims));
                //w = alpha times beta
                SDVariable w = beta.mmul(sd.math.diag(alpha));
                SDVariable b = sd.zero("b_" + r + "_" + layer, 1, numNodes);
                SDVariable adj = sd.constant("A_" + r + "_" + layer, Nd4j.zeros(numNodes, numNodes));
                last = !sigmoid ? w.mmul(last.mmul(adj)).add(b) : sd.nn().sigmoid(w.mmul(last.mmul(adj)).add(b));
            }
        }
        identity = sd.concat(1, last, rt);
        
        //FF layers:
        //1 - inputs: DxN+R, outputs: DxN
        //2 - inputs: DxN, outputs 4x4:
        //3 - inputs: 4x4, outputs 1X2:
        
        
        //DxN+R times W0(N+R, N) = DxN
        //(W2(4, D) times DxN) times W1(N, 4) = 4x4
        // W4(1,4) timex (4x4 times W3(4,2)) = 1x2
        SDVariable w0 = sd.var("cw0", new XavierInitScheme('c', numNodes + numRels, numNodes), DataType.FLOAT, numNodes + numRels, numNodes);
        SDVariable wb0 = sd.zero("cwb0", 1, numNodes);
        
        SDVariable w1 = sd.var("cw1", new XavierInitScheme('c', numNodes, 4), DataType.FLOAT, numNodes, 4);
        SDVariable wb1 = sd.zero("cwb1", 1, 4);
        
        SDVariable w2 = sd.var("cw2", new XavierInitScheme('c', 4, dims), DataType.FLOAT, 4, dims);
        SDVariable wb2 = sd.zero("cwb2", 1, numNodes);
        
        SDVariable w3 = sd.var("cw3", new XavierInitScheme('c', 4, 1), DataType.FLOAT, 4, 2);
        SDVariable w4 = sd.var("cw4", new XavierInitScheme('c', 2, 4), DataType.FLOAT, 1, 4);
        SDVariable wb4 = sd.zero("cwb4", 1, 2);
        
        SDVariable combined = sd.nn.sigmoid(w4.mmul(
                        sd.nn.sigmoid(
                                sd.nn.sigmoid(
                                        sd.nn.sigmoid(
                                                w2.mmul(
                                                        sd.nn.sigmoid(
                                                                identity.mmul(w0).add(wb0)
                                                                )).add(wb2)
                                                ).mmul(w1).add(wb1)
                                        ).mmul(w3)
                                )).add(wb4));
        SDVariable output = sd.nn.softmax("output", combined);
        SDVariable lossForGraph = sd.loss.meanSquaredError(label, combined, null);
        sd.setLossVariables(lossForGraph);

        //Create and set the training configuration
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                //.l2(1e-7) //L2 regularization
                .updater(new Adam(learningRate)) //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input") //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label") //DataSet label array should be associated with variable "label"
                .build();

        sd.setTrainingConfig(config);
        
    }

    /*
    * input is shape [feature index, node index], output is [1,2]
     */
    @Override
    public INDArray output(INDArray input, INDArray relationShipAdjTensor, Triple target) {
        for (int layer = 0; layer < layers; layer++) {
            for (int r = 0; r < numRels; r++) {
                INDArray adjcencyMatrix = relationShipAdjTensor.tensorAlongDimension(r, 1, 2)
                        .castTo(DataType.FLOAT); //self loops;
                INDArray deg = Nd4j.diag(adjcencyMatrix.sum(1));//Transforms.pow(Nd4j.diag(adjcencyMatrix.sum(1)), -.5);
                adjcencyMatrix = adjcencyMatrix.add(Nd4j.eye(numNodes));
                INDArray normalizedadjcencyMatrix = deg.mul(adjcencyMatrix);
                sd.getVariable("A_" + r + "_" + layer).setArray(normalizedadjcencyMatrix);
            }
        }

        INDArray rtArr = Nd4j.zeros(dims, numRels);
        for(int d=0;d<dims;d++)
            rtArr.putScalar(new int[]{d, target.r}, 1);
        sd.getVariable("rt").setArray(rtArr);
        sd.getVariable("input").setArray(input);
        return sd.getVariable("output").eval();
    }

    /*
    * input is shape [feature index, node index], relshipTensor: [rels, nodes, nodes], output is [1,2]
     */
    @Override
    public void fit(INDArray input, INDArray output, INDArray relationShipAdjTensor, Triple target) {
        for (int layer = 0; layer < layers; layer++) {
            for (int r = 0; r < numRels; r++) {
                INDArray adjcencyMatrix = relationShipAdjTensor.tensorAlongDimension(r, 1, 2)
                        .castTo(DataType.FLOAT); //self loops;
                INDArray deg = Nd4j.diag(adjcencyMatrix.sum(1));//Transforms.pow(Nd4j.diag(adjcencyMatrix.sum(1)), -.5);
                adjcencyMatrix = adjcencyMatrix.add(Nd4j.eye(numNodes));
                INDArray normalizedadjcencyMatrix = deg.mul(adjcencyMatrix);
                sd.getVariable("A_" + r + "_" + layer).setArray(normalizedadjcencyMatrix);
            }
        }

        INDArray rtArr = Nd4j.zeros(dims, numRels);
        for(int d=0;d<dims;d++)
            rtArr.putScalar(new int[]{d, target.r}, 1);
        sd.getVariable("rt").setArray(rtArr);
        //sd.getVariable("input").setArray(input);
        //sd.getVariable("label").setArray(output);
        MultiDataSet ds = new MultiDataSet(input, output);
        sd.fit(ds);
    }
}
