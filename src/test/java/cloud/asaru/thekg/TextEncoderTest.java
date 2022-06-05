
package cloud.asaru.thekg;

import java.util.Arrays;
import java.util.Random;
import me.xdrop.fuzzywuzzy.FuzzySearch;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import text.TextEncoder;

/**
 *
 * @author Jae
 */
public class TextEncoderTest {
    
    public static final TextEncoder instance = new TextEncoder(new CharEncoder(360, 30, 50, 0), .20f);
    
    
    public static final String data0 = "But investors with their funds in terra have seen their savings drop by around half";
    
    public static final String data1 = "terra was in the top 10 cryptoassets, with a value of over $18.7 billion";
    public static final String data2 = "terra top cryptoassets, with a value billion";
    public TextEncoderTest() {
    }


    /*@Test
    public void testAutoencode() {
        GraphEncoder instance = new GraphEncoder(6, 4,50, 30, 20);
        for(int i=0;i<1;i++)
            instance.fit(data);
        Triple[] result = instance.autoencode(data);
        assertArrayEquals(data, result);
        result = instance.autoencode(data2);
        assertArrayEquals(data2, result);
    }*/

    @Test
    public void testAutoencodeSmallOther2() {
        
        String result = instance.autoencode(data1);
        assertEquals(data1, result);
    }
    @Test
    public void testAutoencodeSmallOther1() {
        String result1 = instance.autoencode(data1);
        assertTrue(FuzzySearch.ratio(data0, result1) < FuzzySearch.ratio(data2, result1));
    }
    
    
    @Test
    public void testEmbeddingDist() {
        INDArray result0 = instance.embedding(data0);
        INDArray result1 = instance.embedding(data1);
        INDArray result2 = instance.embedding(data2);
        assertTrue(result1.distance2(result0) < result2.distance2(result1));
    }
    
    public final static String corpus="Some cryptocurrencies have always been fairly volatile, with values soaring or plunging within a short space of time. So for the more cautious investor, “stablecoins” were considered the sensible place to go. As the name implies, they are designed to be a steadier and safer bet.\n" +
"\n" +
"At the moment though, that stability is proving hard to find. The value of one of the most popular stablecoins, terra USTUSD, -10.33% (also known as UST), has fluctuated wildly in the last few days, before dropping dramatically—and is yet to recover.\n" +
"\n" +
"Breaking news: Terra blockchain halted\n" +
"\n" +
"Before the crash, terra was in the top 10 cryptoassets, with a value of over $18.7 billion. At the time of writing, this had collapsed to less than $5 billion.\n" +
"\n" +
"In simple terms, the potential for a cryptocurrency crisis is very real.\n" +
"Investors have taken to social media to lament this development. Some spoke of lost life savings and the devastating impact of the currency’s collapse.\n" +
"\n" +
"And they are right to be worried. The impact of volatility in the stablecoin arena should not be underestimated and could destabilise the entire sector.\n" +
"\n" +
"For in theory, stablecoins are supposed to offer the transactional benefits of more traditional cryptoassets (such as bitcoin BTCUSD, +0.80% ) but with a predictably stable worth.\n" +
"\n" +
"Many stablecoins are backed by other assets (typically the U.S. dollar BUXX, +0.45% ) or commodities (often gold GC00, +0.20% ) and involve the stablecoin provider buying—and then holding—the equivalent amount of their chosen asset to ensure the coin remains stable. So while the value of the underlying asset might increase or decrease, the value of the stablecoin should at least remain at a consistent ratio with whatever underpins it.\n" +
"\n" +
"But “algorithmic stablecoins” like terra work differently. Terra holds no reserve asset or commodity, and instead is meant to hold its value using an algorithm, which is designed to maintain a balance between the stablecoin and a partner coin (a more traditional cryptocurrency).\n" +
"\n" +
"Investing Insights with Global Context\n" +
"Understand how today’s global business practices, market dynamics, economic policies and more impact you with real-time news and analysis from MarketWatch.\n" +
"SUBSCRIBE NOW: US $1 FOR 4 WEEKS\n" +
"MarketWatch on Multiple devices\n" +
"In this case terra is tied to a partner coin called Luna LUNAUSD, -8.86 —and the value of Luna has crashed. Its value is now less than $0.02 having been trading at around $82.00 just seven days earlier. In a climate where the value of terra and Luna are both drastically declining, the algorithm cannot solve the issue of decreasing faith in the paired currencies—and the stabilisation feature simply does not work.\n" +
"\n" +
"As a result, fear kicks in and more people sell, just like a traditional bank run, where there is mass withdrawal of funds and sudden drastic loss in value. Stablecoins backed by assets tend to avoid this, due to the long-term steady value of their peg which builds consumer confidence.\n" +
"\n" +
"But they have issues too. Tether USDTUSD, +0.02%, a coin pegged to the U.S. dollar, has had bumps in the road amid questions over whether the company which issues the coins hold the reserves it claims to have. And in recent days tether too has seen its value fall.\n" +
"\n" +
"Why Markets Are Falling So Much\n" +
"YOU MAY ALSO LIKE\n" +
"Why Markets Are Falling So Much\n" +
"Why Markets Are Falling So Much\n" +
"Play video: Why Markets Are Falling So Much\n" +
"Save the savings\n" +
"All of this undermines the basic premise of these coins—that they will remain stable. Customers choose to buy them to either shield against volatility in traditional crypto markets until they rise again, or to use them as a more traditional account (like a regular bank account) and take advantage of the benefits they offer with regard to speed, cost, and ease of international transactions.\n" +
"\n" +
"But investors with their funds in terra have seen their savings drop by around half. The fact it has still not stabilised does little to alleviate worries. In simple terms, the potential for a cryptocurrency crisis is very real.\n" +
"\n" +
"This is why the approach of governments world-wide needs to change. While plenty has been said about regulation in the U.K. and the U.S., there has been little meaningful action.\n" +
"\n" +
"Also read: Cryptocurrency market turbulence is not ‘real threat’ to U.S. financial stability, Yellen says\n" +
"\n" +
"If they fail to act, it will be difficult to advocate the use of stablecoins if they continue to expose consumers to the very volatility and risk they are supposed to avoid.\n" +
"\n" +
"On the Hill: FTX’s Bankman-Fried and CME’s Duffy square off on crypto futures in Capitol Hill hearing\n" +
"\n" +
"The time for allowing the sector freedom to innovate seems to have passed. Regulation is essential—to offer consumer protection, and ban excessively risky practices—if the potential of stablecoins is to be realised. That potential is something that many feel could revolutionise the global economy, speeding up transactions, reducing costs and increasing transparency.\n" +
"\n" +
"But allowing the sector the opportunity to innovate should not come at the expense of people’s savings. If withdrawals persist, it will test both the stability of a particular stablecoin, and more broadly, whether the entire sector has a future. One stablecoin struggling is bad news. But two or more could be catastrophic for customer confidence";
    
    
    static {
        instance.learnWords(corpus);
        instance.trimCommonWords();
        Random rand = new Random(123);
        for(int i=0;i<50000;i++){
            int len = 50 + rand.nextInt(300);
            int start = rand.nextInt(corpus.length() - len);
            String example = corpus.substring(start, start+len);
            //System.out.println(random.toString());
            instance.fit(example);
            if(i%100==0) {
                String result = instance.autoencode(data0);
                System.out.println(result);
                String result2 = instance.autoencode(data1);
                System.out.println(result2);
                System.out.println("");
            }
        }
    }
}
