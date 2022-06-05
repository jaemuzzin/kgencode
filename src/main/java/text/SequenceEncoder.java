package text;

import cloud.asaru.thekg.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.misc.RepeatVector;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Jae
 */
public class SequenceEncoder {
    public MultiLayerNetwork rnet;
    int maxLength;
    int embeddingSize;
    int encoderSize;
    
    /*
    list of vectors
    */
    public void fit(INDArray s) {
        INDArray filterMask = Nd4j.zeros(1, maxLength);
        for(int i=maxLength-(int)s.shape()[0];i<maxLength;i++){
            filterMask.putScalar(new int[]{0, i}, 1);
        }
        rnet.rnnClearPreviousState();
        rnet.fit(Nd4j.expandDims(s.transpose(), 0), Nd4j.expandDims(s.transpose(), 0), filterMask, filterMask);
    }

    public INDArray autoencode(INDArray s) {
        INDArray filterMask = Nd4j.zeros(1, maxLength);
        for(int i=maxLength-(int)s.shape()[0];i<maxLength;i++){
            filterMask.putScalar(new int[]{0, i}, 1);
        }
        rnet.rnnClearPreviousState();
        return rnet.output(Nd4j.expandDims(s.transpose(), 0), false, filterMask, filterMask).tensorAlongDimension(0, 1, 2).transpose();
    }
    public INDArray embedding(INDArray s) {
        INDArray filterMask = Nd4j.zeros(1, maxLength);
        for(int i=maxLength-(int)s.shape()[0];i<maxLength;i++){
            filterMask.putScalar(new int[]{0, i}, 1);
        }
         
        MultiLayerNetwork enet = new TransferLearning.Builder(rnet)
                .removeLayersFromOutput(4).build();
        enet.init();
        return enet.output(Nd4j.expandDims(s.transpose(), 0));
    }
    public SequenceEncoder(int encoderSize, int embeddingSize, int maxLength) {
        this.encoderSize = encoderSize;
        this.embeddingSize = embeddingSize;
        this.maxLength = maxLength;
         MultiLayerConfiguration rconf = new NeuralNetConfiguration.Builder()
                .seed(123)
                 .miniBatch(false)
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp(0.01)).biasInit(0)
                .list()
                .layer(new LSTM.Builder().nIn(encoderSize).nOut((encoderSize+embeddingSize/2)).activation(Activation.TANH).build())
                //.layer(new LSTM.Builder().nOut((encoderSize+embeddingSize/2)).activation(Activation.TANH).build())
                .layer(new LastTimeStep(new LSTM.Builder().nOut(embeddingSize).activation(Activation.TANH).build()))
                .layer(new RepeatVector.Builder(maxLength).build())
                .layer(new LSTM.Builder().nIn(embeddingSize).nOut((encoderSize+embeddingSize/2)).activation(Activation.TANH).build())
                //.layer(new LSTM.Builder().nOut((encoderSize+embeddingSize/2)).activation(Activation.TANH).build())
                .layer(new LSTM.Builder().nOut(encoderSize).activation(Activation.TANH).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID).nIn(encoderSize).nOut(encoderSize).build())
                .build();
         
        rnet = new MultiLayerNetwork(rconf);
        
        rnet.setListeners(new ScoreIterationListener(190));   //Print the score (loss function value) every 20 iterations

        rnet.init();
    }
}
