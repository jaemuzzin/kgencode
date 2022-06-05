package text;

import cloud.asaru.thekg.CharEncoder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Jae
 */
public class TextEncoder {

    private HashMap<String, Integer> commonWords = new HashMap<String, Integer>();
    private HashMap<String, Integer> words = new HashMap<>();
    private HashMap<String, Integer> wordFreqs = new HashMap<>();
    private CharEncoder charEncoder;

    //common words / total vocabulary
    private float commonWordRatio;

    public TextEncoder(CharEncoder charEncoder, float commonWordRatio) {
        this.charEncoder = charEncoder;
        this.commonWordRatio = commonWordRatio;
    }

    public TextEncoder(CharEncoder charEncoder) {
        this.charEncoder = charEncoder;
    }

    public void learnWords(String corpus) {
        String[] wordArr = corpus.split("[^a-zA-Z0-9\\-']");
        Arrays.stream(wordArr)
                .map(w -> w.toLowerCase())
                .map(w -> w.trim())
                .forEach(word -> {
            if (!wordFreqs.containsKey(word)) {
                wordFreqs.put(word, 1);
            }
            wordFreqs.put(word, wordFreqs.get(word) + 1);
        });
    }

    public void trimCommonWords() {
        commonWords.clear();
        words.clear();
        wordFreqs.entrySet().stream().sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
                .limit((int) (commonWordRatio * wordFreqs.size()))
                .forEach(e -> commonWords.put(e.getKey(), commonWords.size() + 1));
        wordFreqs.entrySet().stream().sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
                .skip((int) (commonWordRatio * wordFreqs.size()))
                .forEach(e -> words.put(e.getKey(), words.size() + 1));
    }

    public void fit(String sentence) {
        
        String[] sentenceArr = sentence.split("[^a-zA-Z0-9\\-']");
        
        charEncoder.fit(
        Arrays.stream(sentenceArr)
                .map(w -> w.toLowerCase())
                .map(w -> w.trim())
                .filter(w -> words.containsKey(w))
                .mapToInt(w -> words.get(w))
                .mapToObj(id -> new String(new char[]{(char) id}))
                .reduce((a,b) -> (a==null ? "" : a)+(b==null ? "" : b))
                .get()
        );
    }
    
    private String translateToText(String charoutput){
        return charoutput.chars()
                .filter(c -> words.containsValue(c))
                .mapToObj(c -> words.entrySet().stream().filter(e -> e.getValue()==c).findAny().get().getKey())
                .reduce((a,b) -> (a==null ? "" : a)+" "+(b==null ? "" : b))
                .get();
    }

    public String autoencode(String sentence) {
        String[] sentenceArr = sentence.split("[^a-zA-Z0-9\\-']");
        return translateToText(charEncoder.autoencode( Arrays.stream(sentenceArr)
                .map(w -> w.toLowerCase())
                .map(w -> w.trim())
                .filter(w -> words.containsKey(w))
                .mapToInt(w -> words.get(w))
                .mapToObj(id -> new String(new char[]{(char) id}))
                .reduce((a,b) -> (a==null ? "" : a)+(b==null ? "" : b))
                .get()));
    }

    public INDArray embedding(String sentence) {
        String[] sentenceArr = sentence.split("[^a-zA-Z0-9\\-']");
        return charEncoder.embedding(Arrays.stream(sentenceArr)
                .map(w -> w.toLowerCase())
                .map(w -> w.trim())
                .filter(w -> words.containsKey(w))
                .mapToInt(w -> words.get(w))
                .mapToObj(id -> new String(new char[]{(char) id}))
                .reduce((a,b) -> (a==null ? "" : a)+(b==null ? "" : b))
                .get());
    }

}
