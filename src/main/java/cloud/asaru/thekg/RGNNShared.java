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

/**
 *
 * @author Jae
 */
public class RGNNShared extends RelGNNBuilder implements RelGNN{

    public RGNNShared() {
    }

    @Override
    public RelGNN build(INDArray relationShipAdjTensor, int numNodes, int dims, int layers, boolean learnable, boolean sigmoid, FeatureExtractor featureExtractor) {
        return new RGNNShared(relationShipAdjTensor, numNodes, dims, layers, sigmoid, featureExtractor);
    }

    @Override
    public INDArray getMultiRelAdjacencyTensor() {
        return adjcencyTensor;
    }

    @Override
    public void setMultiRelAdjacencyTensor(INDArray t) {
        if(!t.equalShapes(this.adjcencyTensor)) throw new IllegalArgumentException("Tried to set adjacency matrix to different shape:"+t.shapeInfoToString());
        this.adjcencyTensor = t;
        for (int layer = 0; layer < layers; layer++) {
            for (int r = 0; r < t.shape()[0]; r++) {
                INDArray adjcencyMatrix = t.tensorAlongDimension(r, 1, 2)
                        .castTo(DataType.FLOAT); //self loops;
                INDArray deg = InvertMatrix.invert(Nd4j.diag(adjcencyMatrix.sum(1)), false);//Transforms.pow(Nd4j.diag(adjcencyMatrix.sum(1)), -.5);
                adjcencyMatrix = adjcencyMatrix.add(Nd4j.eye(numNodes));
                INDArray normalizedadjcencyMatrix = deg.mul(adjcencyMatrix);
                sd.constant("A_" + r + "_" + layer, normalizedadjcencyMatrix);
            }
        }
    }
    int numNodes;
    int dims;
    SameDiff sd;
    SDVariable softmax;
    SDVariable identity;
    SDVariable sigmoid;
    SDVariable label;
    INDArray adjcencyTensor;
    int layers;

    protected RGNNShared(INDArray relationShipAdjTensor, int numNodes, int dims, int layers, boolean sigmoid, FeatureExtractor featureExtractor) {
        this.adjcencyTensor = relationShipAdjTensor;
        this.numNodes = numNodes;
        this.layers = layers;
        this.dims = dims;
        sd = SameDiff.create();
        //the shared wieghts
        //Create input and label variables
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, numNodes);
        SDVariable X = featureExtractor.extract(in);
        SDVariable beta = sd.var("beta", new XavierInitScheme('c', dims, dims), DataType.FLOAT, dims, dims);
        label = sd.placeHolder("label", DataType.FLOAT, 2, 1);
        SDVariable last = X;
        for (int layer = 0; layer < layers; layer++) {
            for (int r = 0; r < relationShipAdjTensor.shape()[0]; r++) {
                /* the trained per-relationship linear combination of beta */
                SDVariable alpha = sd.var("alpha_" + r + "_" + layer, Nd4j.ones(dims));
                //w = alpha times beta
                SDVariable w = beta.mmul(sd.math.diag(alpha));
                SDVariable b = sd.zero("b_" + r + "_" + layer, 1, dims);
                INDArray adjcencyMatrix = relationShipAdjTensor.tensorAlongDimension(r, 1, 2)
                        .castTo(DataType.FLOAT); //self loops;
                INDArray deg = InvertMatrix.invert(Nd4j.diag(adjcencyMatrix.sum(1)), false);//Transforms.pow(Nd4j.diag(adjcencyMatrix.sum(1)), -.5);
                adjcencyMatrix = adjcencyMatrix.add(Nd4j.eye(numNodes));
                INDArray normalizedadjcencyMatrix = deg.mul(adjcencyMatrix);
                SDVariable adj = sd.constant("A_" + r + "_" + layer, normalizedadjcencyMatrix);
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
        SDVariable graphProbability = sd.nn().softmax( 
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
                .dataSetFeatureMapping("X") //DataSet features array should be associated with variable "input"
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
    public INDArray output(INDArray input) {
        INDArray r = Nd4j.zeros(input.shape());
        for (int i = 0; i < input.shape()[0]; i++) {
            INDArray inp = input.tensorAlongDimension(i, 1, 2);
            sd.getVariable("X").setArray(inp);
            r.putColumn(i, identity.eval());
        }
        return r;
    }

    /*
    * input is shape [feature index, node index]
     */
    public void fit(INDArray input, INDArray output) {

        for (int i = 0; i < input.shape()[0]; i++) {
            INDArray inp = input.tensorAlongDimension(i, 1);
            INDArray outp = output.tensorAlongDimension(i, 1);
            sd.getVariable("X").setArray(inp);
            DataSet ds = new DataSet(input, outp);
            sd.fit(ds, new ScoreListener(200));
        }
    }
    /*
    * input is shape [feature index, node index]
    * output is shape [2]
     */
    public void fitGraph(INDArray input, INDArray output) {

        for (int i = 0; i < input.shape()[0]; i++) {
            INDArray inp = input.tensorAlongDimension(i, 1);
            INDArray outp = output.tensorAlongDimension(i, 1, 2);
            sd.getVariable("X").setArray(inp);
            DataSet ds = new DataSet(input, outp);
            sd.fit(ds, new ScoreListener(200));
        }
    }
}
