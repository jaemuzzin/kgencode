
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
    
    public final static String corpus="Some cryptocurrencies have always been fairly volatile, with values soaring or plunging within a short space of time. So for the more cautious investor, “stablecoins” were considered the sensible place to go. As the name implies, they are designed to be a steadier and safer bet. " +
 "At the moment though, that stability is proving hard to find. The value of one of the most popular stablecoins, terra USTUSD, -10.33% (also known as UST), has fluctuated wildly in the last few days, before dropping dramatically—and is yet to recover. " +
 "Breaking news: Terra blockchain halted " +
 "Before the crash, terra was in the top 10 cryptoassets, with a value of over $18.7 billion. At the time of writing, this had collapsed to less than $5 billion. " +
 "In simple terms, the potential for a cryptocurrency crisis is very real. " +
"Investors have taken to social media to lament this development. Some spoke of lost life savings and the devastating impact of the currency collapse. " +
 "And they are right to be worried. The impact of volatility in the stablecoin arena should not be underestimated and could destabilise the entire sector. " +
 "For in theory, stablecoins are supposed to offer the transactional benefits of more traditional cryptoassets (such as bitcoin BTCUSD, +0.80% ) but with a predictably stable worth. " +
 "Many stablecoins are backed by other assets (typically the U.S. dollar BUXX, +0.45% ) or commodities (often gold GC00, +0.20% ) and involve the stablecoin provider buying—and then holding—the equivalent amount of their chosen asset to ensure the coin remains stable. So while the value of the underlying asset might increase or decrease, the value of the stablecoin should at least remain at a consistent ratio with whatever underpins it. " +
 "But “algorithmic stablecoins” like terra work differently. Terra holds no reserve asset or commodity, and instead is meant to hold its value using an algorithm, which is designed to maintain a balance between the stablecoin and a partner coin (a more traditional cryptocurrency). " +
 "Investing Insights with Global Context " +
"Understand how today global business practices, market dynamics, economic policies and more impact you with real-time news and analysis from MarketWatch. " +
"SUBSCRIBE NOW: US $1 FOR 4 WEEKS " +
"MarketWatch on Multiple devices " +
"In this case terra is tied to a partner coin called Luna LUNAUSD, -8.86 —and the value of Luna has crashed. Its value is now less than $0.02 having been trading at around $82.00 just seven days earlier. In a climate where the value of terra and Luna are both drastically declining, the algorithm cannot solve the issue of decreasing faith in the paired currencies—and the stabilisation feature simply does not work. " +
 "As a result, fear kicks in and more people sell, just like a traditional bank run, where there is mass withdrawal of funds and sudden drastic loss in value. Stablecoins backed by assets tend to avoid this, due to the long-term steady value of their peg which builds consumer confidence. " +
 "But they have issues too. Tether USDTUSD, +0.02%, a coin pegged to the U.S. dollar, has had bumps in the road amid questions over whether the company which issues the coins hold the reserves it claims to have. And in recent days tether too has seen its value fall. " +
 "Why Markets Are Falling So Much " +
"YOU MAY ALSO LIKE " +
"Why Markets Are Falling So Much " +
"Why Markets Are Falling So Much " +
"Play video: Why Markets Are Falling So Much " +
"Save the savings " +
"All of this undermines the basic premise of these coins—that they will remain stable. Customers choose to buy them to either shield against volatility in traditional crypto markets until they rise again, or to use them as a more traditional account (like a regular bank account) and take advantage of the benefits they offer with regard to speed, cost, and ease of international transactions. " +
 "But investors with their funds in terra have seen their savings drop by around half. The fact it has still not stabilised does little to alleviate worries. In simple terms, the potential for a cryptocurrency crisis is very real. " +
 "This is why the approach of governments world-wide needs to change. While plenty has been said about regulation in the U.K. and the U.S., there has been little meaningful action. " +
 "Also read: Cryptocurrency market turbulence is not ‘real threat’ to U.S. financial stability, Yellen says " +
 "If they fail to act, it will be difficult to advocate the use of stablecoins if they continue to expose consumers to the very volatility and risk they are supposed to avoid. " +
 "On the Hill: FTX Bankman-Fried and CME Duffy square off on crypto futures in Capitol Hill hearing " +
 "The time for allowing the sector freedom to innovate seems to have passed. Regulation is essential—to offer consumer protection, and ban excessively risky practices—if the potential of stablecoins is to be realised. That potential is something that many feel could revolutionise the global economy, speeding up transactions, reducing costs and increasing transparency. " +
 "But allowing the sector the opportunity to innovate should not come at the expense of people savings. If withdrawals persist, it will test both the stability of a particular stablecoin, and more broadly, whether the entire sector has a future. One stablecoin struggling is bad news. But two or more could be catastrophic for customer confidenceHOME " +
"INVESTING " +
"CRYPTOCURRENCY " +
"Are Gemini Layoffs Proof of a ‘Crypto Winter?' " +
"Crypto: 46,000 People Lost $1 Billion to Cons in 15 Months " +
"Scammers are taking advantage of the crypto craze to line their pockets, FTC says. " +
"LUC OLINGA4 HOURS AGO " +
"The euphoria around cryptocurrencies in 2021 has not only made millionaires and billionaires.  " +
 "It was also a ruin for thousands of retail investors.  " +
 "And things are not getting better since the beginning of 2022. Crypto scams, which multiplied last year, continue. " +
 "Since the start of 2021, more than 46,000 people have reported losing over $1 billion in crypto to scams, according to a new report from the Federal Trade Commission (FTC). That about one out of every four dollars reported lost, more than any other payment method.  " +
 "The median individual reported loss is $2,600.  " +
 "The top cryptocurrencies people said they used to pay scammers were Bitcoin (70%), the king of crypto, stablecoin Tether (10%), and Ether (9%), the second-largest crypto by market value.   " +
 "The agency claims that crypto has become \"an alarmingly common method for scammers to get peoples’ money.\"Crypto has several features that are attractive to scammers, which may help to explain why the reported losses in 2021 were nearly sixty times what they were in 2018,\" the FTC said. " +
 "The Median Crypto Loss to Romance Scammers Is $10,000 " +
"Firstly, there is no bank or other centralized authority to flag suspicious transactions and attempt to stop fraud before it happens.  " +
 "Secondly, when the crypto transfer is made it can’t be reversed. " +
 "Thirdly, people are still unfamiliar with how crypto works.  " +
 "Scroll to Continue " +
"TheStreet Recommends " +
"TECHNOLOGY " +
"Musk and Bezos Agree on Who Is Responsible for Inflation " +
"INVESTING " +
"Costco Members Are Not Going to Like This " +
"INVESTING " +
"Las Vegas Strip Casinos Get Some Very Bad News " +
"The report also singles out ads and social media as the perfect pair that makes life easier for scammers. " +
 "\"Nearly half the people who reported losing crypto to a scam since 2021 said it started with an ad, post, or message on a social media platform.\"During this period, nearly four out of every ten dollars reported lost to a fraud originating on social media was lost in crypto, far more than any other payment method. \"In order, the scammers were luckiest on Instagram (32%), followed by Facebook (26%), WhatsApp (9%), and Telegram (7%). " +
 "The most common type of crypto fraud was investment scams, followed by romance -- with $185 million in reported cryptocurrency losses since 2021, which is nearly one in every three dollars reported lost to a romance scam during this period scams -- business imposters, and then government imposters. " +
 "Since 2021, $575 million of all crypto fraud losses reported to the FTC were about bogus investment opportunities.  " +
 "\"The stories people share about these scams describe a perfect storm: false promises of easy money paired with people limited crypto understanding and experience. Investment scammers claim they can quickly and easily get huge returns for investors. But those crypto 'investments' go straight to a scammer wallet,\" the report said. " +
 "As of romance scams, scammers reportedly brag about their supposed wealth and sophistication: \"Before long, they casually offer tips on getting started with crypto investing and help with making investments. The median individual reported crypto loss to romance scammers is $10,000, according to the FTC. " +
"On Saturday, bitcoin (BTC) rose by 0.56%. Partially reversing a 2.50% fall from Friday, bitcoin ended the day at $29,845. " +
 "A bearish start saw bitcoin fall to an early morning low of $29,467 before finding support. " +
 "Steering clear of the day Major Support Levels, bitcoin struck an afternoon intraday high of $29,954. " +
 "Falling short of the First Major Resistance Level at $30,488, however, bitcoin slipped back into the red before a late recovery. " +
 "Saturday upside came despite the US nonfarm payroll figures on Friday, which supported a more aggressive Fed interest rate path trajectory. " +
 "The Bitcoin Fear & Greed Index Sits Deep in the Extreme Fear Zone " +
"Today, the Fear & Greed Index fell from 14/100 to 10/100 despite bitcoin Saturday gain and the prospect of ending a nine-week losing streak. " +
 "While falling deeper into the “Extreme Fear” zone, the Index continued to hold above May low of 8/100. " +
 "Regulatory chatter was market negative, with regulators and lawmakers calling for greater oversight. " +
 "Going into the weekend, Governor Christopher J. Waller talked about “Risk in the Crypto Markets.” " +
 "The governor talked of high volatility being the rule and not the exception and the frequent occurrence of fraud and theft. " +
 "Waller also focused on retail users with a lack of crypto experience and the need for some standard rules. " +
 "South Korean lawmakers were also active going into the weekend. According to local media, regulators plan to move beyond the Capital Markets Act following the collapse of TerraUSD (UST) and Terra LUNA. " +
"Crypto market conditions improved at the turn of the month, as the dust began to settle from the collapse of TerraUSD (UST) and Terra LUNA. " +
 "Conditions improved despite lawmakers and regulators calling for greater oversight to protect retail investors. " +
 "News updates on LUNA 2.0 failed to impress for a second consecutive week. While recovering from a launch day tumble to a low of $4.14, LUNA 2.0 was still down 48.8% to $5.47. " +
 "Launched on Saturday, LUNA 2.0 hit a high of $11.48 before hitting reverse. " +
 "Bitcoin (BTC) at Risk of a Record Tenth Consecutive Weekly Loss " +
"A bullish start to the week saw bitcoin reverse the previous week loss, with a 7.69% rally seeing bitcoin revisit $32,000. " +
 "A bearish Wednesday and investor reaction to US nonfarm payroll numbers on Friday sent bitcoin back to sub-$30,000. " +
 "At the time of writing, bitcoin was 0.94% to $29,754 for the week ending June 5. Bitcoin will need to avoid sub-$29,500 to end a nine-week losing streak. " +
 "BTCUSD 0506 Weekly Chart " +
"BTCUSD 0506 Weekly Chart " +
"Monday through Friday, bitcoin was up 0.77% compared with the NASDAQ 100, which fell by 0.98%. Bitcoin Monday breakout came on a US public holiday, which shielded bitcoin from a closer correlation to the NASDAQ. " +
 "The Crypto Bears Loosened the Grip in a Mixed Week for the Top Ten " +
"In the week ending June 5, SOL is heading for a 12.7% slide to lead the way down, " +
 "BNB (-2.42%), DOGE (-1.55%), and ETH (-0.89%) were also heading for weekly losses. " +
 "Joining bitcoin in positive territory for the week include ADA up 16.8% and XRP (+0.94%). " +
 "The total crypto market cap rose to a Monday high of $1,326 billion before sliding to a Friday low of $1,185 billion. A hold onto $1,210 billion would leave the total market cap flat for the week. " +
 "Avoiding another weekly loss would end a run of eight consecutive weekly declines that started in the week commencing April 4" +
"Bitcoin (BTC) Price Action " +
"At the time of writing, BTC was down 0.45% to $29,712. " +
 "A range-bound start to the day saw bitcoin rise to an early morning high of $29,882 before falling to a low of $29,712. " +
 "BTCUSD 050622 Daily Chart " +
"BTCUSD 050622 Daily Chart " +
"Technical Indicators " +
"BTC will need to move back through the $29,754 pivot to target the First Major Resistance Level at $30,045. " +
 "BTC would need the broader crypto market to support to break out from Saturday high of $29,954. " +
 "An extended rally would test the Second Major Resistance Level at $30,241 and resistance at $30,500. The Third Major Resistance Level sits at $30,728. " +
 "Failure to move back through the pivot would test the First Major Support Level at $29,554. Barring another extended sell-off, BTC should steer clear of sub-$29,000 levels. The Second Major Support Level at $29,267 should limit the" +
"The regulator says here are the things to know to avoid crypto cons: " +
 "Only scammers will guarantee profits or big returns. No cryptocurrency investment is ever guaranteed to make money, let alone big money. " +
"Nobody legit will require you to buy cryptocurrency. Not to sort out a problem, not to protect your money. That a scam. " +
"Never mix online dating and investment advice. If a new love interest wants to show you how to invest in crypto, or asks you to send them crypto, that a scam.";
    
    public static final TextEncoder instance = new TextEncoder(40, 40, 100, corpus);
    
    
    public static final String data0 = "But investors with their funds in terra have seen their savings drop by around half";
    
    public static final String data1 = "terra was in the top 10 cryptoassets, with a value of over $18.7 billion";
    public static final String data2 = "terra top cryptoassets, with a value billion";
    public TextEncoderTest() {
    }


    /*@Test
    public void testAutoencodeSmallOther2() {
        
        String result = instance.autoencode(data1);
        assertEquals(data1, result);
    }
    @Test
    public void testAutoencodeSmallOther1() {
        String result1 = instance.autoencode(data1);
        assertTrue(FuzzySearch.ratio(data0, result1) < FuzzySearch.ratio(data2, result1));
    }*/
    
    
    @Test
    public void testEmbeddingDist() {
        INDArray result0 = instance.embedding(data0);
        INDArray result1 = instance.embedding(data1);
        INDArray result2 = instance.embedding(data2);
        assertTrue(result1.distance2(result0) < result2.distance2(result1));
    }
    
    
    static {
        Random rand = new Random(123);
        for(int i=0;i<50000;i++){
            int len = 50 + rand.nextInt(1000);
            int start = rand.nextInt(corpus.length() - len);
            String example = corpus.substring(start, start+len);
            //System.out.println(random.toString());
            instance.fit(example);
            if(i%50==0) {
                String result = instance.autoencode(data0);
                System.out.println(result);
                String result2 = instance.autoencode(data1);
                System.out.println(result2);
                System.out.println("");
            }
        }
    }
}
