package cloud.asaru.thekg;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.weightinit.impl.XavierInitScheme;

/**
 *
 * @author Jae
 */
public class RGNNSharedOrderInvariant extends RelGNNBuilder implements RelGNN{

    public RGNNSharedOrderInvariant() {
    }

    @Override
    public RelGNN build(INDArray relationShipAdjTensor, int numNodes, int layers, boolean learnable, boolean sigmoid) {
        return new RGNNSharedOrderInvariant(relationShipAdjTensor, numNodes, layers, sigmoid);
    }
    int numNodes;
    SameDiff sd;
    SDVariable softmax;
    SDVariable identity;
    SDVariable sigmoid;
    SDVariable label;

    private RGNNSharedOrderInvariant(INDArray relationShipAdjTensor, int numNodes, int layers, boolean sigmoid) {
        this.numNodes = numNodes;
        sd = SameDiff.create();
        //the shared wieghts
        //Create input and label variables
        SDVariable in = sd.placeHolder("X", DataType.FLOAT, -1, numNodes);
        SDVariable beta = sd.var("beta", new XavierInitScheme('c', numNodes, numNodes), DataType.FLOAT, numNodes, numNodes);
        label = sd.placeHolder("label", DataType.FLOAT, -1, numNodes);
        SDVariable last = in;
        for (int layer = 0; layer < layers; layer++) {
            for (int r = 0; r < relationShipAdjTensor.shape()[0]; r++) {
                /* the trained per-relationship linear combination of beta */
                SDVariable alpha = sd.var("alpha_" + r + "_" + layer, Nd4j.ones(numNodes));
                //w = alpha times beta
                SDVariable w = beta.mmul("w_" + r + "_" + layer, sd.math.diag(alpha));
                SDVariable b = sd.zero("b_" + r + "_" + layer, 1, numNodes);
                INDArray adjcencyMatrix = relationShipAdjTensor.tensorAlongDimension(r, 1, 2)
                        .castTo(DataType.FLOAT); //self loops;
                INDArray deg = InvertMatrix.invert(Nd4j.diag(adjcencyMatrix.sum(1)), false);//Transforms.pow(Nd4j.diag(adjcencyMatrix.sum(1)), -.5);
                adjcencyMatrix = adjcencyMatrix.add(Nd4j.eye(numNodes));
                INDArray normalizedadjcencyMatrix = deg.mul(adjcencyMatrix);
                SDVariable adj = sd.constant("A_" + r + "_" + layer, normalizedadjcencyMatrix);
                last = !sigmoid ? last.mmul(adj).mmul(w).add(b) : sd.nn().sigmoid(last.mmul(adj).mmul(w).add(b));
            }
        }
        identity = last;

        //Define loss function:
        SDVariable loss = sd.math.squaredDifference(identity, label).mean();
        sd.setLossVariables(loss);

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
    * input is shape [feature index, node index], output is same
     */
    @Override
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
    @Override
    public void fit(INDArray input, INDArray output) {

        for (int i = 0; i < input.shape()[0]; i++) {
            INDArray inp = input.tensorAlongDimension(i, 1, 2);
            INDArray outp = output.tensorAlongDimension(i, 1, 2);
            sd.getVariable("X").setArray(inp);
            DataSet ds = new DataSet(input, outp);
            sd.fit(ds, new ScoreListener(200));
        }
    }

}
