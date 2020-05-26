# Differences within Topics

What differences can we see in how men and
women write about topics?

## Male vs. Female Authors

Let's first have a look at the topics and terms distinctive
for men and women.

For women, the topics are:
``` 
    topics                                       dunning    frequency_score    freq both     f women       f men
--  ----------------------------------  ----------------  -----------------  -----------  ----------  ----------
83  (61) Gender and Feminism                 1.20179e+06           0.879186   0.013838    0.041409    0.00569023
82  (46) Family                         450656                     0.753186   0.0166644   0.0346414   0.0113518
81  (32) Body History                   229529                     0.758294   0.00805475  0.0169878   0.00541485
80  (76) Consumption and consumerism    136193                     0.705405   0.00853473  0.0155044   0.00647504
79  (45) Cultural Turn                   68048.6                   0.600953   0.0216482   0.0292282   0.0194082
78  (71) Sexuality                       49462.2                   0.628844   0.00918562  0.0134353   0.00792977
77  (79) Legal History                   47404.4                   0.593071   0.0179887   0.0237402   0.016289
76  (54) Witchcraft and Magic            45443.9                   0.64804    0.00616467  0.00952315  0.00517217
75  (55) U.S. Civil Rights Movement      40766.4                   0.616965   0.009389    0.0132739   0.00824093
74  (40) European Colonization of Asia   32429.1                   0.633953   0.0055189   0.00819067  0.00472934
73  (39) Music                           30484.2                   0.600461   0.00980148  0.0132146   0.00879282
72  (59) Political Economy               28041.6                   0.589033   0.0117095   0.0152735   0.0106563
```

For men, the topics are 
``` 
topics distinctive for Corpus 2: men. 7506 Documents

    topics                                                 dunning    frequency_score    freq both     f women      f men
--  ---------------------------------------------------  ---------  -----------------  -----------  ----------  ---------
 0  (82) German Intellectual History                     -186645             0.250662   0.0151486   0.0059743   0.0178597
 1  (81) General Historiography                          -175105             0.382217   0.0560106   0.0379546   0.0613466
 2  (66) German and Austro-Hungarian Diplomatic History  -166587             0.21653    0.0107333   0.00355286  0.0128553
 3  (25) Narratives and Discourses                       -124987             0.322415   0.0187475   0.0101321   0.0212935
 4  (70) 20th Century British Foreign Policy             -111431             0.290825   0.012415    0.0058829   0.0143454
 5  (26) American Political Thought                      -101144             0.27425    0.00981867  0.00432394  0.0114425
 6  (74) U.S. Political Parties & Elections               -97515.5           0.273308   0.00939561  0.00412003  0.0109547
 7  (64) Political History of the Cold War                -89289.6           0.294453   0.0102683   0.00494225  0.0118422
 8  (29) Nazi Germany                                     -71084.7           0.338276   0.0126514   0.00727906  0.014239
 9  (56) 19th Century U.S. Political History              -66216.9           0.347299   0.0130949   0.00780029  0.0146596
10  (87) Sociology and History                            -61582.4           0.406986   0.0306807   0.022679    0.0330453
11  (33) Political History (Revolutions)                  -44817.3           0.422508   0.0315632   0.0245983   0.0336215
```

The table includes two measurements, dunning for Dunning Log-Likelihood score as well as Frequency score. We will cover them in a moment.
- freq_both: the average weight of the topic in both male and female corpora
- f women: the average weight of the topic in the women corpus
- f men: the average weight of the topic in the men corpus


### Dunning's Log-Likelihood Test
`dunning` is the Dunning Log-Likelihood test (G2) for the value. 

For a longer discussion of this score, see https://de.dariah.eu/tatom/feature_selection.html#chi2

Here's the intuition: Given two text corpora, our null hypothesis is that a given term should appear with the
same frequency in both corpora. e.g. we would expect the word "the" to make up about 1-2% of both corpora. 

Dunning's Log-Likelihood test gives us a way to test how far a term deviates from this null hypothesis, i.e.
how over or under-represented a term is in the female or male corpora (mathematically: how far the observed term counts
deviate from the expected term counts). 

Generally, terms will score high for G2 if a) they are frequent and b) they are heavily skewed towards one or another corpus. 
Ben Schmidt has a blog post which demonstrates how G2 captures a combination of additive difference (difference in absolute numbers) and
multiplicative difference (difference between frequencies): http://sappingattention.blogspot.com/2011/10/comparing-corpuses-by-word-use.html 

A Dunning value of 4 means p value < 0.05. A p value of more than a million as we find for "Gender and Feminism" means many many standard deviations away from expectation. 

### Frequency Score

The frequency score is another measure that we can use to measure how gendered a topic is. It goes from 
0 (only men write about a topic) to 1 (only women write about a topic).<br>
The math behind this score is simple:<br>
`frequency score(topic) = weight of topic among women / (weight of topic among women + weight of 
topic among men)`<br>
Here's an explanation that's more intuitive: Assume that we have an equal number of articles written
by men and women (we don't but let's presume that's the case for the moment). In that case, a
frequency score of 0.5 means that women contributed 50% of the weight to the topic and men the
remaining 50%. A score of 0.89 (the score of "Gender and Feminism") means that women contributed
90% of the weight and men 10%. <br>
The reality is more complicated simply because we have about three times as many articles authored
by men as articles authored by women (7506 to 2016). Hence, men did contribute about one third of 
the "Gender and Feminism" topic weight (37 to 79), simply because even when they are not gender
historians, they will occasionally use "Gender and Feminism" terms like "women," "male," or 
"female." <br>
If we look at the average weights, however, we see that among women, "Gender and 
Feminism" has an average weight of 4.14% whereas among men, it has an average weight of 0.56%. This
is the relationship that the frequency score expresses: 4.14% / (4.14% + 0.56%) = 0.88.

So, in very short, men and women work on different topics... very unexpected.

### Same patterns in dissertations

Note that similar patterns appear in the dissertations. 

The main difference to my mind is that military history is not a top 10 topic for men in 
journals but it is the top topic in the dissertations. 

Women:
``` 
    topics                                         dunning    frequency_score    freq both     f women       f men
--  ------------------------------------------  ----------  -----------------  -----------  ----------  ----------
83  (61) Gender and Feminism                    132238               0.894352   0.0237381   0.0509479   0.00601836
82  (46) Family                                  23348               0.761456   0.0113671   0.0194604   0.00609643
81  (32) Body History                            12211.7             0.737454   0.00739005  0.0121149   0.00431309
80  (45) Cultural Turn                            8433.14            0.601986   0.03108     0.0391042   0.0258544
79  (76) Consumption and consumerism              8340.54            0.68989    0.00825764  0.0123874   0.00556821
```

Men: 
``` 
    topics                                                 dunning    frequency_score    freq both     f women       f men
--  ---------------------------------------------------  ---------  -----------------  -----------  ----------  ----------
 0  (31) Military History                                -21831.7            0.306919   0.0259533   0.0147297   0.0332624
 1  (64) Political History of the Cold War               -12533.4            0.267503   0.0103616   0.00504775  0.0138221
 2  (33) Political History (Revolutions)                  -8365.38           0.430131   0.0726945   0.0607435   0.0804774
 3  (20) Markets & Trade                                  -6502.87           0.401196   0.0286146   0.0220402   0.032896
 4  (70) 20th Century British Foreign Policy              -4254.78           0.300733   0.0047566   0.0026388   0.00613577
```

### Distinctive Terms

We can also look at the most distinctive terms for men and women.

If you needed evidence that without female historians, we wouldn't know about female
historical actors, this is probably useful evidence.

Women: 
``` 
      terms       dunning    frequency_score    count both    c women    c men
----  --------  ---------  -----------------  ------------  ---------  -------
1000  women      85194              0.85247         115376      73315    42061
 999  her        29439.3            0.781682         76056      39492    36564
 998  she        19654.6            0.792389         45845      24535    21310
 997  woman      13445.1            0.828251         22559      13369     9190
 996  female     10980.4            0.828885         18319      10876     7443
 995  family      8903.66           0.697497         56828      23312    33516
 994  gender      8410.04           0.819803         15226       8808     6418
 993  children    7344.82           0.706586         42025      17683    24342
 992  male        5507.25           0.756391         18276       8839     9437
 991  marriage    5144.46           0.738334         20611       9477    11134
 990  sexual      4824.61           0.764324         14778       7308     7470
 989  mother      4103.41           0.762848         12756       6282     6474
```

``` 
terms distinctive for Corpus 2: men. 7506 Documents

    terms         dunning    frequency_score    count both    c women    c men
--  ----------  ---------  -----------------  ------------  ---------  -------
 0  he           -3136.34           0.434877        283376      53388   229988
 1  german       -1982.26           0.357124         40739       5847    34892
 2  his          -1937.44           0.45359         336499      67389   269110
 3  party        -1452.58           0.379307         40835       6356    34479
 4  war          -1380.23           0.427445        101420      18642    82778
 5  marx         -1359.12           0.231291          8857        737     8120
 6  states       -1236.21           0.421788         78725      14199    64526
 7  military     -1200.23           0.376344         32253       4967    27286
 8  roosevelt    -1113.99           0.273057          9845       1002     8843
 9  general      -1085.15           0.412992         56444       9882    46562
10  germany      -1061.68           0.359882         22627       3281    19346
11  philosophy   -1013.13           0.333868         15784       2073    13711
```

Note: of course, the usual caveats (men's corpus skewed towards earlier dates) apply here.
I'm happy to produced decade by decade lists for this analysis.

# Gender Differences in Military History

But for now, let's look at something more interesting: differences within a topic. 

By Dunning score, "Military History" is the 12th most distinctive topic for men. 
So it's not quite at the top but I think within the paper it can stand
rhetorically as one of the most obviously male-coded topics. 

But women have also contributed substantially to the topic. 

What we can do is compare topics that men and women write about ONLY in a dataset of the articles 
that score in the top 10% for the military history topic.

That gives us 134 articles by women and 830 by men.

That gives us the following topics as over-represented among the female-authored articles.
Note that fs_overall means: frequency score in all articles.
e.g. in the overall sample, the women and gender frequency score is 0.88. In the war 
sub-sample, it is 0.90
``` 
    topics                                         dunning    frequency_score    fs_overall    freq both     f women       f men
--  -------------------------------------------  ---------  -----------------  ------------  -----------  ----------  ----------
83  (61) Women and Gender                         97406.9            0.901918      0.879186   0.00873743  0.0350678   0.00381357
82  (46) Family                                   18999.2            0.773872      0.753186   0.00623431  0.0154425   0.00451235
81  (28) Holocaust                                14721.3            0.683299      0.611316   0.0141501   0.0258208   0.0119676
80  (32) Body History                             14612.5            0.833359      0.758294   0.00260167  0.00798053  0.00159581
79  (90) Cultural History of the U.S. Civil War   14082.7            0.690494      0.491617   0.0122915   0.0229678   0.0102951
78  (59) Political Economy                        13657.5            0.700636      0.589033   0.0104491   0.0201914   0.00862729
77  (50) 20th Century Labor History               13375.7            0.71341       0.542426   0.00872338  0.0175884   0.0070656
76  (79) Legal History                            10392.5            0.636187      0.593071   0.0203999   0.031909    0.0182476
75  (40) European Colonization of Asia             8058.22           0.687076      0.633953   0.00736031  0.0135991   0.00619364
74  (43) Islamic History                           7342.91           0.643686      0.593335   0.012714    0.0203786   0.0112806
```

More interesting here are the actual articles behind these topics. 

So, among the 62 articles by women in the the top 5% military history, we're now selecting the 
ones with the highest weight in gender and feminism.
```
   (1990) Susan Zeiger: Finding a Cure for War: Women's Politics and the Peace Movement in the 1920s
   (1994) Philippa Levine: "Walking the Streets in a Way No Decent Woman Should": Women Police in World War I
   (2002) Sharon Block: Rape without Women: Print Culture and the Politicization of Rape, 1765-1815
   (1996) Margaret H. Darrow: French Volunteer Nursing and the Myth of War Experience in World War I
   (1998) Sonya O. Rose: Sex, Citizenship, and the Nation in World War II Britain
```

Family:
``` 
   (2012) Jane Turner Censer: Finding the Southern Family in the Civil War
   (2012) Sarah C. Chambers: The Paternal Obligation to Provide: Political Familialism in Early-Nineteenth-Century Chile
   (1996) Megan J. McClintock: Civil War Pensions and the Reconstruction of Union Families
   (2010) Christina Firpo: Crises Of Whiteness And Empire In Colonial Indochina: The Removal Of Abandoned Eurasian Children From The Vietnamese Milieu, 1890-1956
   (2005) Tammy M. Proctor: Family Ties in the Making of Modern Intelligence
```

Holocaust:
``` 
   (2009) Kirsten Wächter; Judith Keilbach: Photographs, Symbolic Images, and the Holocaust: On the (IM)Possibility of Depicting Historical Truth
   (1997) Marita Sturken: Reenactment, Fantasy, and the Paranoia of History: Oliver Stone's Docudramas
   (1998) Julia A. Thomas: Photography, National Identity, and the "Cataract of Times": Wartime Images and the Case of Japan
   (2001) Denise J. Youngblood: A War Remembered: Soviet Films of the Great Patriotic War
   (2008) Lara Deeb: Exhibiting the "Just-Lived Past": Hizbullah's Nationalist Narratives in Transnational Political Context
```


Body History:
``` 
   (1979) D. Maier: Nineteenth-Century Asante Medical Practices
   (2007) Frances Clarke: So Lonesome I Could Die: Nostalgia and Debates over Emotional Control in the Civil War North
   (2001) Cheryl A. Wells: Battle Time: Gender, Modernity, and Confederate Hospitals
   (2003) Darlene Clark Hine: Black Professionals and Race Consciousness: Origins of the Civil Rights Movement, 1890-1950
   (1996) Margaret H. Darrow: French Volunteer Nursing and the Myth of War Experience in World War I
```

Cultural History of the U.S. Civil War:
``` 
   (2007) Kate Masur: "A Rare Phenomenon of Philological Vegetation": The Word "Contraband" and the Meanings of Emancipation in the United States
   (1999) Alice Fahs: The Feminized Civil War: Gender, Northern Popular Literature, and the Memory of the War, 1861-1900
   (1990) Drew Gilpin Faust: Altars of Sacrifice: Confederate Women and the Narratives of War
   (2007) Frances Clarke: So Lonesome I Could Die: Nostalgia and Debates over Emotional Control in the Civil War North
   (1996) Megan J. McClintock: Civil War Pensions and the Reconstruction of Union Families
``` 

Political Economy:
```
   (1990) Susan Pedersen: Gender, Welfare, and Citizenship in Britain during the Great War
   (1996) Megan J. McClintock: Civil War Pensions and the Reconstruction of Union Families
   (2003) Margot Canaday: Building a Straight State: Sexuality and Social Citizenship under the 1944 G.I. Bill
   (2006) Elna C. Green: Protecting Confederate Soldiers and Mothers: Pensions, Gender, and the Welfare State in the U.S. South, a Case Study from Florida
   (2001) Jacqueline Jones: Federal Power, Southern Power: A Long View, 1860-1940


```


The data for the dissertations looks very similar:
``` 
topics distinctive for Corpus 1: women. 169 Documents

    topics                               dunning    frequency_score    fs_comp_to_overall    freq both     f women        f men
--  ---------------------------------  ---------  -----------------  --------------------  -----------  ----------  -----------
83  (61) Gender and Feminism            4707.4             0.930632             0.0514459   0.00973312  0.0436208   0.00325142
82  (32) Body History                    709.345           0.835329             0.0770351   0.0038118   0.0116912   0.00230471
81  (45) Cultural Turn                   469.819           0.699646             0.0986925   0.0111437   0.0213921   0.00918353
80  (39) Music                           335.222           0.764497             0.164035    0.00372721  0.00889232  0.00273928
79  (76) Consumption and consumerism     330.561           0.846427             0.141022    0.00158784  0.00507517  0.000920821
```

## Nazi History

``` 

topics distinctive for Corpus 1: women. 119 Documents

    topics                                       dunning    frequency_score    fs_overall    freq both     f women        f men
--  -----------------------------------------  ---------  -----------------  ------------  -----------  ----------  -----------
83  (61) Women and Gender                       83002.8            0.885244      0.879186  0.00908895   0.036037    0.00467154
82  (28) Holocaust                              17004.1            0.713715      0.611316  0.0119317    0.0245779   0.00985868
81  (54) Witchcraft and Magic                   16556.2            0.750052      0.64804   0.00755377   0.0176843   0.00589313
80  (77) Classics                               10767.8            0.670738      0.421338  0.0134099    0.0238358   0.0117009
79  (47) Progressive Era to New Deal             9452.17           0.69508       0.488343  0.00841545   0.0162543   0.00713048
78  (30) Urban History                           9418.58           0.737488      0.557972  0.00496557   0.0111171   0.00395719
77  (70) 20th Century British Foreign Policy     7316.55           0.713504      0.290825  0.00514705   0.0105946   0.00425408
76  (39) Music                                   6589.12           0.632287      0.600461  0.015128     0.0236194   0.0137361
75  (14) Colonies and Empires                    5850.03           0.641046      0.570176  0.0115537    0.0185772   0.0104023
74  (9) Natural Resources and the Environment    4795.62           0.864825      0.409526  0.000648723  0.00235791  0.000368549


topics distinctive for Corpus 2: men. 838 Documents

    topics                                                 dunning    frequency_score    fs_overall    freq both      f women       f men
--  ---------------------------------------------------  ---------  -----------------  ------------  -----------  -----------  ----------
 0  (82) German Intellectual History                     -28074.9           0.281299       0.250662   0.0442048   0.0189237    0.048349
 1  (25) Narratives and Discourses                       -23359.6           0.216264       0.322415   0.0234561   0.00720746   0.0261196
 2  (10) Nutrition and Climate                           -15915.5           0.0377944      0.46098    0.0066467   0.000301928  0.00768675
 3  (66) German and Austro-Hungarian Diplomatic History  -15380.5           0.333942       0.21653    0.0392809   0.0211818    0.0422477
 4  (58) Soviet Union                                     -7495.63          0.278136       0.472853   0.011513    0.00485642   0.0126041
 5  (67) Italian Renaissance                              -5918.92          0.199596       0.529866   0.00538857  0.00150259   0.00602557
 6  (31) Military History                                 -5783.94          0.336296       0.381601   0.0151529   0.00825116   0.0162842
 7  (75) Agricultural History                             -4991.93          0.30247        0.513642   0.00939029  0.00442486   0.0102042
 8  (33) Political History (Revolutions)                  -4604.73          0.423009       0.422508   0.0478856   0.0364773    0.0497557
 9  (57) Demography                                       -4464.13          0.371661       0.414266   0.0180978   0.0113583    0.0192026

```



## Gender differences in articles mentioning gender at least 10 times

If we are interested in particular in gender history as opposed to women's history,
we can create a corpus of articles that mention gender at least 10 times and then compare male
and female authors within that. 

Sorted by frequency score, that gives us:
```
terms distinctive for Corpus 1: male. 93 Documents

     terms          dunning    frequency_score    count both    c male    c female    freq both      f male     f female
---  -----------  ---------  -----------------  ------------  --------  ----------  -----------  ----------  -----------
226  gay           112.439            0.733779           446       276         170  0.000919626  0.00153525  0.000557003
225  masculinity   195.755            0.705865          1033       605         428  0.00212998   0.00336532  0.00140234
224  americans     118.557            0.670073           951       518         433  0.00196091   0.00288138  0.00141872
223  him           193.019            0.669816          1554       846         708  0.00320426   0.00470588  0.00231975
222  service       125.784            0.668166          1034       561         473  0.00213205   0.00312057  0.00154978
221  black         352.21             0.667399          2926      1585        1341  0.00603324   0.00881658  0.00439377
220  he            507.627            0.637385          6446      3279        3167  0.0132913    0.0182395   0.0103766
219  united        136.109            0.629315          1964       982         982  0.00404965   0.00546238  0.00321751
218  his           502.384            0.622519          8132      4007        4125  0.0167677    0.022289    0.0135155
217  war           220.302            0.620546          3689      1810        1879  0.0076065    0.0100681   0.00615652

``` 

For women:
``` 
terms distinctive for Corpus 2: female. 174 Documents

    terms          dunning    frequency_score    count both    c male    c female    freq both       f male    f female
--  ----------  ----------  -----------------  ------------  --------  ----------  -----------  -----------  ----------
 0  feminist     -620.737            0.154205          1588       154        1434   0.00327436  0.000856626  0.00469848
 1  feminism     -385.292            0.180666          1157       133        1024   0.00238567  0.000739814  0.00335512
 2  chinese      -414.325            0.192507          1340       165        1175   0.002763    0.000917814  0.00384987
 3  welfare      -111.759            0.311054           933       196         737   0.00192379  0.00109025   0.00241477
 4  movement     -174.986            0.32987           1784       401        1383   0.0036785   0.00223057   0.00453138
 5  women       -2556.36             0.340928         29584      6909       22675   0.0610005   0.0384314    0.0742943
 6  mothers       -90.0651           0.347158          1132       270         862   0.00233412  0.00150188   0.00282433
 7  rights        -92.1136           0.374559          1695       442        1253   0.00349499  0.00245863   0.00410544
 8  woman        -207.181            0.382797          4339      1161        3178   0.00894676  0.00645807   0.0104127
 9  home          -99.1389           0.384447          2138       575        1563   0.00440843  0.00319844   0.00512115

```

Sample articles:
``` 
Sample articles of distinctive terms for male

 Term: gay. Highest scoring items:
   Count gay: 78. (2007) Craig M. Loftin: Unacceptable Mannerisms: Gender Anxieties, Homosexual Activism, and Swish in the United States, 1945-1965
   Count gay: 51. (1985) George Chauncey: Christian Brotherhood or Sexual Perversion? Homosexual Identities and the Construction of Sexual Boundaries in the World War One Era
   Count gay: 47. (1988) Martin Bauml Duberman: Reclaiming the Gay Past
   Count gay: 35. (2005) Daniel Wickberg: Heterosexual White Male: Some Recent Inversions in American Cultural History
   Count gay: 14. (2000) K. A. Cuordileone: "Politics in an Age of Anxiety": Cold War Political Culture and the Crisis in American Masculinity, 1949-1960

 Term: masculinity. Highest scoring items:
   Count masculinity: 58. (2007) Robert A. Nye: Western Masculinities in War and Peace
   Count masculinity: 55. (2004) Joseph S. Alter: Indian Clubs and Colonialism: Hindu Masculinity and Muscular Christianity
   Count masculinity: 54. (2000) K. A. Cuordileone: "Politics in an Age of Anxiety": Cold War Political Culture and the Crisis in American Masculinity, 1949-1960
   Count masculinity: 48. (1996) Thomas Miller Klubock: Working-Class Masculinity, Middle-Class Morality, and Labor Politics in the Chilean Copper Mines
   Count masculinity: 38. (2005) Edward Behrend-Martínez: Manhood and the Neutered Body in Early Modern Spain

```

# Women and gender, 1970-1989 vs 1990-2009

Let's take all articles scoring in the top 10% for the "women and gender" topic and compare
those from 1970-1989 against those from 1990-2009

``` 
terms distinctive for Corpus 1: 1970-1989. 275 Documents

     terms         dunning    frequency_score    count both    c 1970-1989    c 1990-2009    freq both    f 1970-1989    f 1990-2009
---  ----------  ---------  -----------------  ------------  -------------  -------------  -----------  -------------  -------------
999  women        2418.98            0.589137         85537          29738          55799  0.0344142       0.0441551     0.0307937
998  work          887.506           0.615883         17841           6662          11179  0.00717798      0.00989176    0.00616933
997  family        861.978           0.620263         15981           6037           9944  0.00642965      0.00896376    0.00548778
996  percent       791.419           0.690134          5233           2370           2863  0.0021054       0.00351898    0.00158
995  medical       606.697           0.700549          3539           1646           1893  0.00142385      0.00244399    0.00104469
994  age           512.319           0.651735          5675           2328           3347  0.00228323      0.00345662    0.0018471
993  table         508.519           0.730145          2132           1069           1063  0.000857769     0.00158725    0.000586636
992  married       455.388           0.657279          4652           1936           2716  0.00187164      0.00287458    0.00149887
991  data          403.643           0.71566           1980            957           1023  0.000796615     0.00142096    0.000564561
990  household     383.555           0.662425          3642           1536           2106  0.00146529      0.00228066    0.00116223

terms distinctive for Corpus 2: 1990-2009. 650 Documents

    terms         dunning    frequency_score    count both    c 1970-1989    c 1990-2009    freq both    f 1970-1989    f 1990-2009
--  ----------  ---------  -----------------  ------------  -------------  -------------  -----------  -------------  -------------
 0  gender      -1534.86            0.255389         10233           1157           9076  0.00411705     0.00171792     0.00500876
 1  white       -1046.45            0.281373          8580           1090           7490  0.003452       0.00161844     0.00413349
 2  african      -864.924           0.214873          4354            402           3952  0.00175175     0.000596891    0.00218098
 3  race         -863.09            0.194219          3819            314           3505  0.0015365      0.000466228    0.0019343
 4  black        -782.424           0.300862          7622           1051           6571  0.00306657     0.00156053     0.00362633
 5  war          -714.754           0.336578         10041           1593           8448  0.0040398      0.00236529     0.00466218
 6  racial       -688.777           0.161658          2527            169           2358  0.00101669     0.000250932    0.00130131
 7  nation       -557.4             0.23694           3257            337           2920  0.00131039     0.000500379    0.00161146
 8  british      -470.494           0.268411          3475            417           3058  0.0013981      0.000619163    0.00168761
 9  colonial     -451.788           0.295532          4196            566           3630  0.00168818     0.000840399    0.00200328
```

Unsurprisingly, articles mentioning "white" a lot are about race: 
``` 
 Term: white. Highest scoring items:
   Count white: 232. (2000) Walter Johnson: The Slave Trader, the White Slave, and the Politics of Racial Determination in the 1850s
   Count white: 203. (1999) Martha Mabie Gardner: Working on White Womanhood: White Working Women in the San Francisco Anti-Chinese Movement, 1877-1890
   Count white: 196. (2000) Aline Helg: Black Men, Racial Stereotyping, and Violence in the U.S. South and Cuba at the Turn of the Century
   Count white: 185. (2001) Mara L. Keire: The Vice Trust: A Reinterpretation of the White Slavery Scare in the United States, 1907-1917
   Count white: 180. (2005) Jamie J. Fader; Mark J. Stern; Michael B. Katz: The New African American Inequality

```
