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
    public RelGNN build(SameDiff sd, SDVariable input, int numRels, int numNodes, int dims, int layers, boolean learnable, boolean sigmoid, FeatureExtractor featureExtractor) {
        return new RGNNShared(sd, input, numRels, numNodes, dims, layers, sigmoid, featureExtractor);
    }

    @Override
    public RelGNN build(SameDiff sd, int numRels, int numNodes, int dims, int layers, boolean learnable, boolean sigmoid, FeatureExtractor featureExtractor) {
        return new RGNNShared(sd, numRels, numNodes, dims, layers, sigmoid, featureExtractor);
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
    protected RGNNShared(SameDiff sd, SDVariable in, int numRels, int numNodes, int dims, int layers, boolean sigmoid, FeatureExtractor featureExtractor) {
        init(sd, in, numRels, numNodes, dims, layers, sigmoid, featureExtractor);
    }

    protected RGNNShared(SameDiff sd, int numRels, int numNodes, int dims, int layers, boolean sigmoid, FeatureExtractor featureExtractor) {
        init(sd, sd.placeHolder("input", DataType.FLOAT, dims, numNodes), numRels, numNodes, dims, layers, sigmoid, featureExtractor);
    }

    private void init(SameDiff sd, SDVariable in, int numRels, int numNodes, int dims, int layers, boolean sigmoid, FeatureExtractor featureExtractor) {
        this.numRels = numRels;
        this.numNodes = numNodes;
        this.layers = layers;
        this.dims = dims;
        this.sd = sd;
        //the shared wieghts
        //one-hot vector of relationship identity
        SDVariable rt = sd.constant("rt", Nd4j.zeros(dims, numRels));
        SDVariable X = featureExtractor.extract(in);
        SDVariable beta = sd.var("beta", new XavierInitScheme('c', dims, dims), DataType.FLOAT, dims, dims);
        label = sd.placeHolder("label", DataType.FLOAT, 1, 2);
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
        //head and tail vectors
        identity = sd.concat(1, sd.slice(last, new int[]{0, 0}, dims, 2), last.mean(true, 1), rt);

        //FF layers:
        //first condense dimensions (L0 on left), then features (R0 on right)
        //
        SDVariable l0 = sd.var("L0", new XavierInitScheme('c', dims, dims), DataType.FLOAT, dims, dims);
        SDVariable wb0 = sd.zero("cwb0", dims, 1);
        SDVariable l1 = sd.var("L1", new XavierInitScheme('c', dims, dims), DataType.FLOAT, dims, dims);
        SDVariable wb1 = sd.zero("cwb1", dims, 1);
        SDVariable l2 = sd.var("L2", new XavierInitScheme('c', 1, dims), DataType.FLOAT, 1, dims);

        SDVariable r0 = sd.var("R0", new XavierInitScheme('c', 3 + numRels, 3 + numRels), DataType.FLOAT, 3 + numRels, 3 + numRels);
        SDVariable rb0 = sd.zero("rb0", 1, 3 + numRels);
        SDVariable r1 = sd.var("R1", new XavierInitScheme('c', 3 + numRels, 3 + numRels), DataType.FLOAT, 3 + numRels, 3 + numRels);
        SDVariable rb1 = sd.zero("rb1", 1, 3 + numRels);
        SDVariable r2 = sd.var("R2", new XavierInitScheme('c', 3 + numRels, 2), DataType.FLOAT, 3 + numRels, 2);
        SDVariable rb2 = sd.zero("rb2", 1, 2);
        SDVariable combined
                = sd.nn.sigmoid(
                        sd.nn.sigmoid(
                                sd.nn.sigmoid(
                                        l2.mmul(
                                                sd.nn.sigmoid(
                                                        l1.mmul(
                                                                sd.nn.sigmoid(
                                                                        l0.mmul(identity).add(wb0)
                                                                )
                                                        ).add(wb1)
                                                )
                                        )
                                ).mmul(r0).add(rb0)
                        ).mmul(r1).add(rb1)
                ).mmul(r2).add(rb2);
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
        for (int d = 0; d < dims; d++) {
            rtArr.putScalar(new int[]{d, target.r}, 1);
        }
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
        for (int d = 0; d < dims; d++) {
            rtArr.putScalar(new int[]{d, target.r}, 1);
        }
        sd.getVariable("rt").setArray(rtArr);
        //sd.getVariable("input").setArray(input);
        //sd.getVariable("label").setArray(output);
        MultiDataSet ds = new MultiDataSet(input, output);
        sd.fit(ds);
    }
}
