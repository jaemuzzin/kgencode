package cloud.asaru.thekg;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author Jae
 */
public class RGNN extends RelGNNBuilder implements RelGNN {
    boolean sigmoid;
    MultiLayerConfiguration conff;
    int numNodes;
MultiLayerNetwork nn;

    public RGNN() {
    }

    @Override
    public RelGNN build(INDArray relationShipAdjTensor, int numNodes, int layers, boolean learnable, boolean sigmoid) {
        return new RGNN(relationShipAdjTensor, numNodes, layers, learnable, sigmoid);
    }
    private RGNN(INDArray relationShipAdjTensor, int numNodes, int layers, boolean learnable, boolean sigmoid) {
this.sigmoid = sigmoid;
        this.numNodes = numNodes;
        ListBuilder conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Nadam())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list();
        for (int k = 0; k < layers; k++) {
            for (int r = 0; r < relationShipAdjTensor.shape()[0]; r++) {
                INDArray adjcencyMatrix = relationShipAdjTensor.tensorAlongDimension(r, 1, 2);
                INDArray deg = Transforms.pow(Nd4j.diag(adjcencyMatrix.sum(1)), -.5);
                INDArray normalizedadjcencyMatrix = deg.mul(adjcencyMatrix).mul(deg);
                conf = conf.layer(new FrozenLayerWithBackprop(new DenseLayer.Builder()
                        .nIn(numNodes).nOut(numNodes)
                        .hasBias(false)
                        .hasLayerNorm(false)
                        .activation(Activation.IDENTITY)
                        .weightInit(new IWeightInit() {
                            @Override
                            public INDArray init(double fanin, double fanout, long[] shape, char order, INDArray paramView) {
                                INDArray ret;
                                if (order == Nd4j.order()) {
                                    ret = Nd4j.createUninitialized(normalizedadjcencyMatrix.shape()).assign(normalizedadjcencyMatrix);
                                } else {
                                    ret = Nd4j.createUninitialized(shape, order).assign(normalizedadjcencyMatrix);
                                }

                                INDArray flat = Nd4j.toFlattened(order, ret);
                                paramView.assign(flat);
                                return paramView.reshape(order, shape);
                            }
                        })
                        .build()));
                if (learnable) {
                    conf = conf.layer(new DenseLayer.Builder()
                            .nIn(numNodes).nOut(numNodes)
                            .weightInit(WeightInit.XAVIER)
                            .activation(sigmoid ? Activation.SIGMOID : Activation.IDENTITY).build());
                }
            }
        }
        conf.layer(new FrozenLayerWithBackprop(new OutputLayer.Builder()
                .nIn(3).nOut(3)
                .weightInit(WeightInit.IDENTITY)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .hasBias(false).activation(Activation.IDENTITY).build()));
        conff = conf.build();
         nn = new MultiLayerNetwork(conff);
        nn.setListeners(new ScoreIterationListener(100));   //Print the score (loss function value) every 20 iterations

        nn.init();
    }

    /*
    * input is shape [feature index, node index], output is same
     */
    @Override
    public INDArray output(INDArray input) {
        nn.setListeners(new ScoreIterationListener(190));   //Print the score (loss function value) every 20 iterations

        return nn.output(Nd4j.createFromArray(new double[][]{input.toDoubleVector()}));
    }
    
    /*
    * input is shape [feature index, node index]
     */
    @Override
    public void fit(INDArray input, INDArray output) {
        nn.fit(input, output);
    }

}
