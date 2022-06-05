package text;

import org.deeplearning4j.models.word2vec.Word2Vec;
import cloud.asaru.thekg.CharEncoder;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.stream.Collectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Jae
 */
public class TextEncoder {

    //private HashMap<String, Integer> wordFreqs = new HashMap<>();
    private SequenceEncoder wordEncoder;

    //common words / total vocabulary
    Word2Vec word2vec;

    public TextEncoder(int wordDimensions, int encodingDimensions, int maxLength, String corpus) {
        SentenceIterator iter = new BasicLineIterator(new ByteArrayInputStream(corpus.getBytes()));
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        wordEncoder = new SequenceEncoder(wordDimensions, encodingDimensions, maxLength);
        /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
         */
        t.setTokenPreProcessor(new CommonPreprocessor());
        word2vec = new Word2Vec.Builder()
                .minWordFrequency(3)
                .iterations(1)
                .layerSize(wordDimensions)
                .seed(123)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();
        word2vec.fit();

    }

    public void fit(String sentence) {

        String[] sentenceArr = sentence.split("[^a-zA-Z0-9\\-']");
        INDArray sentenceVecs = Nd4j.concat(0, (INDArray[]) Arrays.stream(sentenceArr)
                .map(w -> w.toLowerCase())
                .map(w -> w.trim())
                .filter(w -> word2vec.hasWord(w))
                .map(w -> word2vec.getWordVectorMatrix(w))
                .collect(Collectors.toList()).toArray(new INDArray[0]));
        if (sentenceVecs.shape()[0] > 0) {
            wordEncoder.fit(sentenceVecs);
        }
    }

    private String translateToText(INDArray arrOutput) {
        StringBuilder out = new StringBuilder();
        for (long wi = 0; wi < arrOutput.shape()[0]; wi++) {
            out.append(word2vec.wordsNearestSum(Nd4j.expandDims(arrOutput, 0), 1).iterator().next());
            out.append(" ");
        }
        return out.toString();
    }

    public String autoencode(String sentence) {
        String[] sentenceArr = sentence.split("[^a-zA-Z0-9\\-']");
        INDArray sentenceVecs = Nd4j.concat(0, (INDArray[]) Arrays.stream(sentenceArr)
                .map(w -> w.toLowerCase())
                .map(w -> w.trim())
                .filter(w -> word2vec.hasWord(w))
                .map(w -> word2vec.getWordVectorMatrix(w))
                .collect(Collectors.toList()).toArray(new INDArray[0]));
        return translateToText(wordEncoder.autoencode(sentenceVecs));
    }

    public INDArray embedding(String sentence) {
        String[] sentenceArr = sentence.split("[^a-zA-Z0-9\\-']");
        INDArray sentenceVecs = Nd4j.concat(0, (INDArray[]) Arrays.stream(sentenceArr)
                .map(w -> w.toLowerCase())
                .map(w -> w.trim())
                .filter(w -> word2vec.hasWord(w))
                .map(w -> word2vec.getWordVectorMatrix(w))
                .collect(Collectors.toList()).toArray(new INDArray[0]));
        return wordEncoder.embedding(sentenceVecs);
    }

}
