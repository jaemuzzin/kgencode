package cloud.asaru.thekg;

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
public class CharEncoder {
    public MultiLayerNetwork rnet;
    int maxLength;
    int embeddingSize;
    int encoderSize;
    int startCharacter;
    private INDArray prep(String s){
        if(s.length() > maxLength-1) s = s.substring(0, maxLength);
        int[][][] timeseries = new int[1][encoderSize][maxLength];
        for(int i = maxLength-s.length(); i<maxLength; i++){
            //get to 0
            for(int j=0;j < encoderSize;j++) {
                timeseries[0][j][i] = 0;
            }
            timeseries[0][s.charAt(i - (maxLength-s.length()))-startCharacter][i] = 1;
        }
        INDArray input = Nd4j.createFromArray(timeseries);
        return input;
    }
    private String unprep(INDArray h, INDArray mask){
        StringBuilder d = new StringBuilder();
        for(int i=0;i<maxLength;i++){
            if(mask.getInt(0, i)==1){
                int hindex=0;
                double max= 0;
                for(int j=0;j<encoderSize;j++) {
                    if(h.getDouble(0, j, i) > max){
                        max = h.getDouble(0, j, i);
                        hindex=j;
                    }
                }
                d.append((char)(hindex+startCharacter));
            }
        }
        return d.toString();
    }
    public void fit(String s) {
        INDArray input = prep(s);
        INDArray filterMask = Nd4j.zeros(1, maxLength);
        for(int i=maxLength-s.length();i<maxLength;i++){
            filterMask.putScalar(new int[]{0, i}, 1);
        }
        rnet.rnnClearPreviousState();
        rnet.fit(input, input, filterMask, filterMask);
    }

    public String autoencode(String s) {
        INDArray filterMask = Nd4j.zeros(1, maxLength);
        for(int i=maxLength-s.length();i<maxLength;i++){
            filterMask.putScalar(new int[]{0, i}, 1);
        }
        rnet.rnnClearPreviousState();
        return unprep(rnet.output(prep(s)), filterMask);
    }
    public INDArray embedding(String s) {
        INDArray filterMask = Nd4j.zeros(1, maxLength);
        for(int i=maxLength-s.length();i<maxLength;i++){
            filterMask.putScalar(new int[]{0, i}, 1);
        }
         
        MultiLayerNetwork enet = new TransferLearning.Builder(rnet)
                .removeLayersFromOutput(4).build();
        enet.init();
        return enet.output(prep(s));
    }
    public CharEncoder(int encoderSize, int embeddingSize, int maxLength, int startCharacter) {
        this.encoderSize = encoderSize;
        this.embeddingSize = embeddingSize;
        this.maxLength = maxLength;
        this.startCharacter=startCharacter;
         MultiLayerConfiguration rconf = new NeuralNetConfiguration.Builder()
                .seed(123)
                 .miniBatch(false)
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp(0.001)).biasInit(0)
                .list()
                .layer(new LSTM.Builder().nIn(encoderSize).nOut((encoderSize+embeddingSize/2)).activation(Activation.TANH).build())
                //.layer(new LSTM.Builder().nOut((encoderSize+embeddingSize/2)).activation(Activation.TANH).build())
                .layer(new LastTimeStep(new LSTM.Builder().nOut(embeddingSize).activation(Activation.TANH).build()))
                .layer(new RepeatVector.Builder(maxLength).build())
                .layer(new LSTM.Builder().nIn(embeddingSize).nOut((encoderSize+embeddingSize/2)).activation(Activation.TANH).build())
                //.layer(new LSTM.Builder().nOut((encoderSize+embeddingSize/2)).activation(Activation.TANH).build())
                .layer(new LSTM.Builder().nOut(encoderSize).activation(Activation.TANH).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(encoderSize).nOut(encoderSize).build())
                .build();
         
        rnet = new MultiLayerNetwork(rconf);
        
        rnet.setListeners(new ScoreIterationListener(190));   //Print the score (loss function value) every 20 iterations

        rnet.init();
    }
}
