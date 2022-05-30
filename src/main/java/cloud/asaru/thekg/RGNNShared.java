package cloud.asaru.thekg;

import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
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
    int numRels;
    int layers;

    protected RGNNShared(int numRels, int numNodes, int dims, int layers, boolean sigmoid, FeatureExtractor featureExtractor) {
        this.numRels = numRels;
        this.numNodes = numNodes;
        this.layers = layers;
        this.dims = dims;
        sd = SameDiff.create();
        //the shared wieghts
        //Create input and label variables
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, numRels, numNodes);
        SDVariable X = featureExtractor.extract(in);
        SDVariable beta = sd.var("beta", new XavierInitScheme('c', dims, dims), DataType.FLOAT, dims, dims);
        label = sd.placeHolder("label", DataType.FLOAT, 2, 1);
        SDVariable last = X;
        for (int layer = 0; layer < layers; layer++) {
            for (int r = 0; r < numRels; r++) {
                /* the trained per-relationship linear combination of beta */
                SDVariable alpha = sd.var("alpha_" + r + "_" + layer, Nd4j.ones(dims));
                //w = alpha times beta
                SDVariable w = beta.mmul(sd.math.diag(alpha));
                SDVariable b = sd.zero("b_" + r + "_" + layer, 1, dims);
                SDVariable adj = sd.constant("A_" + r + "_" + layer, Nd4j.zeros(numNodes, numNodes));
                last = !sigmoid ? w.add(b).mmul(last.mmul(adj)) : sd.nn().sigmoid(w.add(b).mmul(last.mmul(adj)));
            }
        }
        identity = last;
        //DxN times Nx2 = (L1=Dx2)
        SDVariable graphCombinerLayer1 = sd.var("wg1", new XavierInitScheme('c', numNodes, 2), DataType.FLOAT, numNodes, 2);
        SDVariable graphCombinerBias1 = sd.zero("wgb1", numNodes, 1);
        //2xD times Dx2(L1) = 2x2
        SDVariable graphCombinerLayer2 = sd.var("wg2", new XavierInitScheme('c', 2, dims), DataType.FLOAT, 2, dims);
        SDVariable graphCombinerBias2 = sd.zero("wgb2", 2, 1);
        //2x2 times 2x1 = 2x1
        SDVariable graphCombinerLayer3 = sd.var("wg3", new XavierInitScheme('c', 2, 1), DataType.FLOAT, 2, 1);
        SDVariable graphCombinerBias3 = sd.zero("wgb3", 2, 1);
        SDVariable graphProbability = sd.nn().softmax("output",
                //(L2x(XxL1))xL3
                graphCombinerLayer2.add(graphCombinerBias2).mmul(
                        identity.mmul(graphCombinerLayer1.add(graphCombinerBias1))
                ).mmul(
                        graphCombinerLayer3.add(graphCombinerBias3)
                )
        );
        //Define loss function:
        SDVariable lossForGraph = sd.loss.softmaxCrossEntropy(label, graphProbability, null);
        sd.setLossVariables(lossForGraph, lossForGraph);

        //Create and set the training configuration
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .l2(1e-4) //L2 regularization
                .updater(new Adam(learningRate)) //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input") //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label") //DataSet label array should be associated with variable "label"
                .build();

        sd.setTrainingConfig(config);
        /*
        int batchSize = 32;
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator testData = new MnistDataSetIterator(batchSize, false, 12345);

        //Perform training for 2 epochs
        int numEpochs = 2;
        sd.fit(trainData, numEpochs);

        //Evaluate on test set:
        String outputVariable = "softmax";
        Evaluation evaluation = new Evaluation();
        sd.evaluate(testData, outputVariable, evaluation);

        //Print evaluation statistics:
        System.out.println(evaluation.stats());

        //Save the trained network for inference - FlatBuffers format
        File saveFileForInference = new File("sameDiffExampleInference.fb");
        try {
            sd.asFlatFile(saveFileForInference);
        } catch (IOException ex) {
            Logger.getLogger(RGNNShared.class.getName()).log(Level.SEVERE, null, ex);
        }

        SameDiff loadedForInference;
        loadedForInference = SameDiff.fromFlatFile(saveFileForInference);

        //Perform inference on restored network
        INDArray example = new MnistDataSetIterator(1, false, 12345).next().getFeatures();
        loadedForInference.getVariable("input").setArray(example);
        INDArray output = loadedForInference.getVariable("softmax").eval();

        System.out.println("-----------------------");
        System.out.println(example.reshape(28, 28));
        System.out.println("Output probabilities: " + output);
        System.out.println("Predicted class: " + output.argMax().getInt(0));
         */
    }

    /*
    * input is shape [feature index, node index], output is [2,1]
     */
    @Override
    public INDArray output(INDArray input, INDArray relationShipAdjTensor) {
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
        sd.getVariable("input").setArray(input);
        return sd.getVariable("output").eval();
    }

    /*
    * input is shape [feature index, node index], relshipTensor: [rels, nodes, nodes], output is [2,1]
     */
    @Override
    public void fit(INDArray input, INDArray output, INDArray relationShipAdjTensor) {
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

        sd.getVariable("input").setArray(input);
        sd.getVariable("label").setArray(output);
        MultiDataSet ds = new MultiDataSet(input, output);
        sd.fit(ds, new ScoreListener(20));
    }
}
