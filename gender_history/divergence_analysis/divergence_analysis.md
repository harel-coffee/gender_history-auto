

```python
from gender_history.datasets.dataset import Dataset
from gender_history.divergence_analysis.divergence_analysis import divergence_analysis
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
```


<style>.container { width:100% !important; }</style>


# Divergence Analysis
In this notebook, we will look at divergent uses of terms and topics between two corpora. 

What the two corpora are is flexible. We can compare:
<ul>
    <li>Male vs. Female Authors</li>
    <li>1990s vs. 2000s</li>
    <li>Male authors with female advisors vs. male authors with male advisors</li>
    <li>Harvard authors vs. non-Harvard authors</li>
</ul>

In each case, we want to know what terms are over or under-represented in one of the corpora.

## Male vs. Female Authors
Let's make this concrete and first look at male vs. female authors.


```python
# Loads the entire dataset of dissertations
d = Dataset()

# Create two sub-datasets, one for female authors and one for male authors
c1 = d.copy().filter(author_gender='female')
c2 = d.copy().filter(author_gender='male')

# Run the divergence analysis
divergence_analysis(d, c1, c2, c1_name='female author', c2_name='male author',
                    topics_or_terms='terms', number_of_terms_to_print=20)
pass
```

                 female author    female author freq    male author    male author freq
    ---------  ---------------  --------------------  -------------  ------------------
    1976-1984              989             0.114348            2136           0.164497
    1985-1989              950             0.109839            1698           0.130766
    1990-1994             1183             0.136779            1854           0.14278
    1995-1999             1696             0.196092            2340           0.180208
    2000-2004             1762             0.203723            2323           0.178899
    2005-2009             1445             0.167071            1922           0.148017
    2010-2015              624             0.0721471            712           0.0548325
    
    
    Terms distinctive for Corpus 1: female author. 8649 Theses
    
          term         dunning    frequency_score    count_total    count female author    count male author
    ----  ---------  ---------  -----------------  -------------  ---------------------  -------------------
    7834  woman      15294.6             0.918063          15206                  13427                 1779
    7833  gender      2145.67            0.85441            3279                   2617                  662
    7832  female      1925.66            0.90598            2075                   1798                  277
    7831  women       1842.44            0.920505           1804                   1599                  205
    7830  feminist    1177.7             0.945566            978                    901                   77
    7829  family      1036.8             0.713735           5083                   3186                 1897
    7828  child        918.202           0.775819           2545                   1781                  764
    7827  feminism     549.101           0.940041            474                    433                   41
    7826  male         500.911           0.778167           1362                    957                  405
    7825  mother       453.589           0.848799            722                    571                  151
    7824  gendered     449.948           0.901385            501                    431                   70
    7823  sexual       424.104           0.803818            940                    690                  250
    7822  suffrage     399.386           0.883526            501                    419                   82
    7821  home         386.836           0.698111           2238                   1363                  875
    7820  wife         370.289           0.846563            599                    472                  127
    7819  girl         365.21            0.882634            461                    385                   76
    7818  womanhood    320.107           0.958092            246                    231                   15
    7817  marriage     314.909           0.739763           1198                    787                  411
    7816  widow        314.591           0.919234            312                    276                   36
    7815  sex          289.262           0.783541            753                    534                  219
    
    
    Terms distinctive for Corpus 2: male author. 12985 Theses
    
        term             dunning    frequency_score    count_total    count female author    count male author
    --  -------------  ---------  -----------------  -------------  ---------------------  -------------------
     0  army            -661.414          0.255293            3113                    584                 2529
     1  war             -511.006          0.398332           13446                   4147                 9299
     2  military        -436.253          0.330615            4226                   1055                 3171
     3  party           -198.603          0.380496            3810                   1115                 2695
     4  officer         -188.659          0.272829            1031                    208                  823
     5  policy          -184.812          0.417635            7359                   2397                 4962
     6  air             -176.968          0.197333             549                     78                  471
     7  doctrine        -169.192          0.289408            1073                    231                  842
     8  force           -168.064          0.378197            3107                    903                 2204
     9  navy            -165.417          0.236735             677                    117                  560
    10  command         -153.799          0.173765             411                     51                  360
    11  german          -139.904          0.389703            3142                    945                 2197
    12  united_states   -137.841          0.374807            2416                    695                 1721
    13  warfare         -137.259          0.255408             650                    122                  528
    14  general         -132.386          0.387166            2844                    849                 1995
    15  thought         -128.201          0.369073            2059                    582                 1477
    16  fleet           -123.401          0.0701867            186                      9                  177
    17  union           -120.408          0.398117            3160                    974                 2186
    18  operation       -116.37           0.321776            1024                    248                  776
    19  germany         -115.906          0.381634            2268                    666                 1602


Many numbers. Let's start at the top:
```
             female author    female author freq    male author    male author freq
---------  ---------------  --------------------  -------------  ------------------
1976-1984              989             0.114348            2136           0.164497
1985-1989              950             0.109839            1698           0.130766
1990-1994             1183             0.136779            1854           0.14278
1995-1999             1696             0.196092            2340           0.180208
2000-2004             1762             0.203723            2323           0.178899
2005-2009             1445             0.167071            1922           0.148017
2010-2015              624             0.0721471            712           0.0548325
```
This tells us how many theses we have overall and how they are distributed. 

For instance, 11.4% of the female dissertations date from 1976 to 1984 while 16.4% of the male ones do.

This chart can be useful to identify differences that arise from different temporal distributions. e.g. cultural history became more popular during the 2000s. Given that more women wrote their dissertations in the 2000s than in the 1980s, we would expect that cultural history skews female. However, that is in part because the female dataset skews towards later dates.


Let's next look at the 10 terms most distinctive for men and women:
```
Terms distinctive for Corpus 1: female author. 8649 Theses

      term         dunning    frequency_score    count_total    count female author    count male author
----  ---------  ---------  -----------------  -------------  ---------------------  -------------------
7834  woman      15294.6             0.918063          15206                  13427                 1779
7833  gender      2145.67            0.85441            3279                   2617                  662
7832  female      1925.66            0.90598            2075                   1798                  277
7831  women       1842.44            0.920505           1804                   1599                  205
7830  feminist    1177.7             0.945566            978                    901                   77
7829  family      1036.8             0.713735           5083                   3186                 1897
7828  child        918.202           0.775819           2545                   1781                  764
7827  feminism     549.101           0.940041            474                    433                   41
7826  male         500.911           0.778167           1362                    957                  405
7825  mother       453.589           0.848799            722                    571                  151


Terms distinctive for Corpus 2: male author. 12985 Theses

    term             dunning    frequency_score    count_total    count female author    count male author
--  -------------  ---------  -----------------  -------------  ---------------------  -------------------
 0  army           -661.414           0.255293            3113                    584                 2529
 1  war            -511.006           0.398332           13446                   4147                 9299
 2  military       -436.253           0.330615            4226                   1055                 3171
 3  party          -198.603           0.380496            3810                   1115                 2695
 4  officer        -188.659           0.272829            1031                    208                  823
 5  policy         -184.812           0.417635            7359                   2397                 4962
 6  air            -176.968           0.197333             549                     78                  471
 7  doctrine       -169.192           0.289408            1073                    231                  842
 8  force          -168.064           0.378197            3107                    903                 2204
 9  navy           -165.417           0.236735             677                    117                  560
```

### Dunning's Log-Likelihood Test
The tables are sorted by `dunning`, which is the Dunning Log-Likelihood test (G2) for the value. 

For a longer discussion of this score, see https://de.dariah.eu/tatom/feature_selection.html#chi2

Here's the intuition: Given two text corpora, our null hypothesis is that a given term should appear with the
same frequency in both corpora. e.g. we would expect the word "the" to make up about 1-2% of both corpora. 

Dunning's Log-Likelihood test gives us a way to test how far a term deviates from this null hypothesis, i.e.
how over or under-represented a term is in the female or male corpora (mathematically: how far the observed term counts
deviate from the expected term counts). 

Generally, terms will score high for G2 if a) they are frequent and b) they are heavily skewed towards one or another corpus. 
Ben Schmidt has a blog post which demonstrates how G2 captures a combination of additive difference (difference in absolute numbers) and
multiplicative difference (difference between frequencies): http://sappingattention.blogspot.com/2011/10/comparing-corpuses-by-word-use.html 

A Dunning value of 4 means p value < 0.05. A p value of 15314 as we find for "woman" means many many standard deviations away from expectation. 


This leads us to the first observation: The terms associated with female authors are on the whole far more distinctive than those associated with male authors. 
"Woman," "gender," and "female" are all far more strongly associated with female authors than "army," "war," or "military" are with men. Put a bit more strongly:
If we want to know what we wouldn't know with out female historians, these terms would be strong bets.



### Frequency Score
The frequency score corroborates this finding. Assuming that male and female corpora had the same number of words, it indicates what percentage of times a term would appear 
in the female corpus. In this case, "woman" has a frequency score of 91.8%, which means that 91.8% of all mentions of "woman" come from female authors (assuming the same number of 
terms in the male and female corpora). 

Mathematically, the frequency score for a given word w is frequency_corpus1(w) / (frequency_corpus1(w) + frequency_corpus2(w))


This "assuming the same number of words in both corpora" business is more confusing than it needs to be. The point is simply that corpus 1 might be 10 times larger than corpus 2 and
to account for that fact, we're using frequencies instead of absolute counts.


With the frequency score, we find the same pattern as with Dunning: The terms that are most distinctive for female authors skew much stronger / score much higher than those associated
with men. "woman" has a frequency score of 0.92, which means that female authors use "woman" about 10 times more frequently than men. The scores for feminist and feminism are even more extreme at 0.94.
By comparison men use "army" about 3 times more often than women.

## What would we not know about without female historians?
The frequency score gives us a useful tool to figure out what we would not know without female historians. 

In another paper, I have used it to identify rhetorical taboos, i.e. what are terms that one side (in that paper: tobacco industry lawyers) avoid
saying at all costs? 

Here, we can ask a similar question: What are topics that men or women avoid at all costs? (or, turned more positively, what are topics that almost exclusively men or women work on?



```python
divergence_analysis(d, c1, c2, c1_name='female author', c2_name='male author',
                        topics_or_terms='terms', sort_by='frequency_score',
                        number_of_terms_to_print=80, min_appearances_per_term=20)
pass
```

                 female author    female author freq    male author    male author freq
    ---------  ---------------  --------------------  -------------  ------------------
    1976-1984              989             0.114348            2136           0.164497
    1985-1989              950             0.109839            1698           0.130766
    1990-1994             1183             0.136779            1854           0.14278
    1995-1999             1696             0.196092            2340           0.180208
    2000-2004             1762             0.203723            2323           0.178899
    2005-2009             1445             0.167071            1922           0.148017
    2010-2015              624             0.0721471            712           0.0548325
    
    
    Terms distinctive for Corpus 1: female author. 8649 Theses
    
          term                       dunning    frequency_score    count_total    count female author    count male author
    ----  ----------------------  ----------  -----------------  -------------  ---------------------  -------------------
    8678  home_economics            114.393            1                    67                     67                    0
    8677  beguine                    80.4467           1                    48                     48                    0
    8676  midwifery                 102.428            0.98959              65                     64                    1
    8675  rodeo                      97.1513           0.989084             62                     61                    1
    8674  ywca                      182.014            0.982887            119                    116                    3
    8673  alice                     103.877            0.97234              74                     71                    3
    8672  nurse                     263.444            0.964703            194                    184                   10
    8671  black_woman               177.055            0.963669            132                    125                    7
    8670  heroine                    68.3108           0.961174             53                     50                    3
    8669  womanhood                 320.255            0.958115            246                    231                   15
    8668  parlor                     61.6555           0.95794              49                     46                    3
    8667  woman_suffrage             82.4699           0.95772              65                     61                    4
    8666  suffragist                181.654            0.956428            142                    133                    9
    8665  sarah                      93.3123           0.954118             75                     70                    5
    8664  american_women             74.173            0.954118             60                     56                    4
    8663  birth_control             107.483            0.9525               87                     81                    6
    8662  clubwomen                  83.4026           0.950032             69                     64                    5
    8661  costume                    95.9336           0.948239             80                     74                    6
    8660  feminist                 1178.27             0.945595            978                    901                   77
    8659  feminism                  549.373            0.940073            474                    433                   41
    8658  abbess                     51.366            0.939746             46                     42                    4
    8657  motherhood                200.104            0.93729             177                    161                   16
    8656  courtship                  59.0831           0.935719             54                     49                    5
    8655  femininity                165.342            0.934739            149                    135                   14
    8654  cosmetic                   45.0038           0.933823             42                     38                    4
    8653  obscenity                  52.7425           0.930402             50                     45                    5
    8652  unmarried                  52.7425           0.930402             50                     45                    5
    8651  feminine                  247.517            0.930088            229                    206                   23
    8650  sisterhood                 60.4834           0.927918             58                     52                    6
    8649  widowhood                  60.4834           0.927918             58                     52                    6
    8648  louise                     48.0391           0.9258               47                     42                    5
    8647  baby                       94.5962           0.924127             92                     82                   10
    8646  helen                      55.7858           0.923841             55                     49                    6
    8645  actress                    71.2794           0.921243             71                     63                    8
    8644  women                    1843.42             0.920546           1804                   1599                  205
    8643  widow                     314.759            0.919275            312                    276                   36
    8642  gender_relations           51.1324           0.919275             52                     46                    6
    8641  woman                   15302.8              0.918105          15206                  13427                 1779
    8640  midwife                    99.1595           0.916781            101                     89                   12
    8639  nursing                   204.516            0.913562            211                    185                   26
    8638  reproductive              122.417            0.912262            128                    112                   16
    8637  girls                      68.216            0.912262             72                     63                    9
    8636  cancer                     72.8987           0.909914             78                     68                   10
    8635  nun                       260.373            0.908623            277                    241                   36
    8634  working_women              40.4794           0.906146             45                     39                    6
    8633  female                   1926.74             0.906028           2075                   1798                  277
    8632  wctu                       38.9849           0.903914             44                     38                    6
    8631  wear                       38.9849           0.903914             44                     38                    6
    8630  katherine                  38.9849           0.903914             44                     38                    6
    8629  young_woman                51.4034           0.902757             58                     50                    8
    8628  anna                       70.0358           0.90179              79                     68                   11
    8627  gendered                  450.205            0.901435            501                    431                   70
    8626  salon                      73.253            0.899114             84                     72                   12
    8625  rape                      135.361            0.899114            154                    132                   22
    8624  hair                       59.3496           0.897579             69                     59                   10
    8623  lady                      157.431            0.895639            183                    156                   27
    8622  orphanage                  62.5913           0.894815             74                     63                   11
    8621  kindergarten               44.0027           0.893107             53                     45                    8
    8620  mothers                    37.8158           0.89219              46                     39                    7
    8619  maternity                  36.3639           0.889666             45                     38                    7
    8618  dress                     182.541            0.884629            228                    191                   37
    8617  ballet                     33.4866           0.884246             43                     36                    7
    8616  rochester                  33.4866           0.884246             43                     36                    7
    8615  sewing                     33.4866           0.884246             43                     36                    7
    8614  suffrage                  399.63             0.883584            501                    419                   82
    8613  girl                      365.434            0.882692            461                    385                   76
    8612  daughters                  46.2227           0.881331             60                     50                   10
    8611  women_and_men              65.1167           0.881331             84                     70                   14
    8610  domesticity               172.367            0.880763            221                    184                   37
    8609  convent                   144.009            0.880653            185                    154                   31
    8608  pregnancy                  40.0776           0.878961             53                     44                    9
    8607  white_woman                78.2815           0.876494            104                     86                   18
    8606  pornography                59.4074           0.875038             80                     66                   14
    8605  daughter                  206.527            0.87233             280                    230                   50
    8604  sisters                    85.3392           0.871634            117                     96                   21
    8603  african_american_women     47.1722           0.869862             66                     54                   12
    8602  dorothy                    36.3974           0.864628             53                     43                   10
    8601  maternal                   76.9384           0.864238            111                     90                   21
    8600  margaret                   85.4783           0.862086            125                    101                   24
    8599  dancer                     38.3359           0.861333             57                     46                   11
    
    
    Terms distinctive for Corpus 2: male author. 12985 Theses
    
        term                         dunning    frequency_score    count_total    count female author    count male author
    --  -------------------------  ---------  -----------------  -------------  ---------------------  -------------------
     0  cossacks                    -38.5138          0                     44                      0                   44
     1  malcolm                     -40.0683          0.0283006             52                      1                   51
     2  battleship                  -38.1631          0.0294217             50                      1                   49
     3  reconnaissance              -35.3171          0.0312805             47                      1                   46
     4  niebuhr                     -31.5471          0.0341578             43                      1                   42
     5  iww                         -34.7955          0.0560825             52                      2                   50
     6  marine_corps                -27.5517          0.0660592             44                      2                   42
     7  spinoza                     -27.5517          0.0660592             44                      2                   42
     8  ss                         -100.135           0.0677818            150                      7                  143
     9  fleet                      -123.327           0.0702233            186                      9                  177
    10  air_force                   -64.1203          0.0725086            100                      5                   95
    11  armored                     -33.7666          0.0775565             56                      3                   53
    12  wellington                  -30.2416          0.0833599             52                      3                   49
    13  artillery                   -62.2033          0.084147             103                      6                   97
    14  infantry                   -109.312           0.0876844            181                     11                  170
    15  jeffersonian                -37.3645          0.0887561             65                      4                   61
    16  kentuckians                 -26.7642          0.0901021             48                      3                   45
    17  thailand                    -24.1923          0.0959206             45                      3                   42
    18  hegel                       -58.7351          0.0968137            104                      7                   97
    19  guerrilla                   -72.9827          0.0994899            130                      9                  121
    20  brigade                     -39.2848          0.099787              72                      5                   67
    21  newman                      -21.6561          0.102542              42                      3                   39
    22  telegraph                   -42.9759          0.106209              81                      6                   75
    23  cyprus                      -27.0701          0.108142              53                      4                   49
    24  baseball                    -91.3602          0.114388             175                     14                  161
    25  vermont                     -40.806           0.120345              83                      7                   76
    26  serbia                      -20.4887          0.126572              45                      4                   41
    27  stabilization               -36.6787          0.127738              78                      7                   71
    28  apologetic                  -24.265           0.131619              54                      5                   49
    29  enlistment                  -24.265           0.131619              54                      5                   49
    30  armstrong                   -18.89            0.132204              43                      4                   39
    31  salisbury                   -18.89            0.132204              43                      4                   39
    32  blockade                    -43.4022          0.134533              95                      9                   86
    33  apologist                   -18.0989          0.135213              42                      4                   38
    34  tank                        -46.3654          0.13772              103                     10                   93
    35  veto                        -21.8764          0.139009              51                      5                   46
    36  wesley                      -35.596           0.139992              81                      8                   73
    37  football                    -48.5299          0.141661             110                     11                   99
    38  unionists                   -39.3734          0.141661              90                      9                   81
    39  flying                      -20.3079          0.144415              49                      5                   44
    40  foe                         -24.0827          0.146312              58                      6                   52
    41  knox                        -18.7602          0.150259              47                      5                   42
    42  retarded                    -18.7602          0.150259              47                      5                   42
    43  logistics                   -22.5289          0.151279              56                      6                   50
    44  ontology                    -17.9948          0.153362              46                      5                   41
    45  bavarian                    -41.3962          0.15365              101                     11                   90
    46  admiral                     -33.0688          0.154782              82                      9                   73
    47  fortification               -44.3903          0.155232             109                     12                   97
    48  ulama                       -31.5233          0.158452              80                      9                   71
    49  underpinnings               -20.2341          0.159396              53                      6                   47
    50  eisenhower_administration   -16.4822          0.159968              44                      5                   39
    51  aircraft                    -51.0536          0.162299             130                     15                  115
    52  federalist                  -42.7517          0.164605             111                     13                   98
    53  fighter                     -35.2287          0.16615               93                     11                   82
    54  plateau                     -14.9957          0.16717               42                      5                   37
    55  populists                   -23.9755          0.17004               66                      8                   58
    56  nigerian                    -17.2499          0.171678              49                      6                   43
    57  logistical                  -20.24            0.172152              57                      7                   50
    58  command                    -153.671           0.173845             411                     51                  360
    59  landing                     -16.5188          0.17505               48                      6                   42
    60  pacification                -16.5188          0.17505               48                      6                   42
    61  korean_war                  -18.0505          0.18115               54                      7                   47
    62  descartes                   -15.0765          0.182208              46                      6                   40
    63  bomber                      -20.3069          0.183144              61                      8                   53
    64  pact                        -20.3069          0.183144              61                      8                   53
    65  satellite                   -20.3069          0.183144              61                      8                   53
    66  princeton                   -25.5429          0.18371               76                     10                   66
    67  leo                         -17.3323          0.184362              53                      7                   46
    68  heidegger                   -36.0224          0.184362             106                     14                   92
    69  united_states_army          -22.5632          0.184726              68                      9                   59
    70  arnold                      -16.6203          0.18769               52                      7                   45
    71  kant                        -24.1005          0.18837               74                     10                   64
    72  jacksonian                  -44.3174          0.188636             133                     18                  115
    73  bombing                     -39.08            0.188969             118                     16                  102
    74  populism                    -41.3363          0.189501             125                     17                  108
    75  federalists                 -21.1305          0.189976              66                      9                   57
    76  isolationist                -15.9148          0.19114               51                      7                   44
    77  manchuria                   -12.9672          0.194114              43                      6                   37
    78  mistake                     -12.9672          0.194114              43                      6                   37
    79  tang                        -15.2159          0.19472               50                      7                   43


```
Terms distinctive for Corpus 1: female author. 8649 Theses

      term                       dunning    frequency_score    count_total    count female author    count male author
----  ----------------------  ----------  -----------------  -------------  ---------------------  -------------------
8678  home_economics            114.393            1                    67                     67                    0
8677  beguine                    80.4467           1                    48                     48                    0  # Christian order
8676  midwifery                 102.428            0.98959              65                     64                    1
8675  rodeo                      97.1513           0.989084             62                     61                    1
8674  ywca                      182.014            0.982887            119                    116                    3
8673  alice                     103.877            0.97234              74                     71                    3  # refers to many different alices
8672  nurse                     263.444            0.964703            194                    184                   10
8671  black_woman               177.055            0.963669            132                    125                    7
8670  heroine                    68.3108           0.961174             53                     50                    3
8669  womanhood                 320.255            0.958115            246                    231                   15
8668  parlor                     61.6555           0.95794              49                     46                    3
8667  woman_suffrage             82.4699           0.95772              65                     61                    4
8666  suffragist                181.654            0.956428            142                    133                    9
8665  sarah                      93.3123           0.954118             75                     70                    5  # refers to many different sarahs
8664  american_women             74.173            0.954118             60                     56                    4
8663  birth_control             107.483            0.9525               87                     81                    6
8662  clubwomen                  83.4026           0.950032             69                     64                    5
8661  costume                    95.9336           0.948239             80                     74                    6
8660  feminist                 1178.27             0.945595            978                    901                   77
8659  feminism                  549.373            0.940073            474                    433                   41
8658  abbess                     51.366            0.939746             46                     42                    4
8657  motherhood                200.104            0.93729             177                    161                   16
8656  courtship                  59.0831           0.935719             54                     49                    5
8655  femininity                165.342            0.934739            149                    135                   14
8654  cosmetic                   45.0038           0.933823             42                     38                    4
8653  obscenity                  52.7425           0.930402             50                     45                    5
8652  unmarried                  52.7425           0.930402             50                     45                    5
8651  feminine                  247.517            0.930088            229                    206                   23
8650  sisterhood                 60.4834           0.927918             58                     52                    6
8649  widowhood                  60.4834           0.927918             58                     52                    6
8648  louise                     48.0391           0.9258               47                     42                    5
8647  baby                       94.5962           0.924127             92                     82                   10
8646  helen                      55.7858           0.923841             55                     49                    6
8645  actress                    71.2794           0.921243             71                     63                    8
8644  women                    1843.42             0.920546           1804                   1599                  205
8643  widow                     314.759            0.919275            312                    276                   36
8642  gender_relations           51.1324           0.919275             52                     46                    6
8641  woman                   15302.8              0.918105          15206                  13427                 1779
8640  midwife                    99.1595           0.916781            101                     89                   12
8639  nursing                   204.516            0.913562            211                    185                   26
8638  reproductive              122.417            0.912262            128                    112                   16
8637  girls                      68.216            0.912262             72                     63                    9
8636  cancer                     72.8987           0.909914             78                     68                   10

```

This list is suggestive. It includes the terms with a frequency score > 0.909, i.e. they appear at least 10 times more
often in womens' dissertations than in mens'. 

Of the 7 dissertation abstracts that mention "rodeo" only one was written by a man. I guess rodeo provides such a potent starting point
to look at gender, masculinity, performativity, or visual culture that we find primarily women writing about it in our dataset.


```python
d.print_examples_of_term_in_context('rodeo')
```

    
     Found 7 examples of rodeo.
    1998 male    "Times were not easy": A history of New Mexico ranching and its cultur and efficiency in the ranching business rodeo also reflected the modernization of ranching what had 
    1997 female  The American cowgirl: History and iconography, 1860-present            ic cowgirl in dime novel wild_west show rodeo cinema fiction television pornography and advertisemen
    2002 female  Riding pretty: Rodeo royalty in the American West, 1910--1956          riding pretty rodeo royalty in the american west 1910 1956 originally the 
    2009 female  Race, gender, and cultural identity in the American rodeo              r and cultural identity in the american rodeo although western movie and wild_west_shows strove to p
    2008 female  More than Barbie and big hair: A "bling blingin'" visual analysis of w sis of woman and ethnic minority in the rodeo arena this thesis used visual_rhetoric and analytic_in
    2003 female  The legacy of "Six-Shooter Sal" in southeastern Idaho: Historical reco  reputation a tough guy she competed in rodeo worked a ranch hand and won place in local history a f
    1982 female  FROM BUFFALO BILL TO BIG BUSINESS: A STUDY OF FACTORS IN THE EVOLUTION ss study of factors in the evolution of rodeo and the professional rodeo cowboy rodeo an outgrowth o


Some problems:

We're really dealing with a small selected set of texts, i.e. only the abstracts. 
For example, "cancer" makes the list above. We might be tempted to claim that almost only women write about cancer. 
However, if I'm thinking of books on the history of cancer, I'm thinking of Robert Proctor, Allan Brandt, Keith Waillo, and Robin Scheffler, all men.
This can certainly be a result of biases in who I read but at any rate, we couldn't claim that men don't work on the history of cancer.


Incidentally, here's the list of topics covered almost exclusively by men. Again, it is worth noting that the list is much
shorter, i.e. we would lose more coverage if all female historians suddenly disappeared than if all male historians suddenly 
disappeared. 
```
Terms distinctive for Corpus 2: male author. 12985 Theses

    term                         dunning    frequency_score    count_total    count female author    count male author
--  -------------------------  ---------  -----------------  -------------  ---------------------  -------------------
 0  cossacks                    -38.5138          0                     44                      0                   44
 1  malcolm                     -40.0683          0.0283006             52                      1                   51
 2  battleship                  -38.1631          0.0294217             50                      1                   49
 3  reconnaissance              -35.3171          0.0312805             47                      1                   46
 4  niebuhr                     -31.5471          0.0341578             43                      1                   42
 5  iww                         -34.7955          0.0560825             52                      2                   50
 6  marine_corps                -27.5517          0.0660592             44                      2                   42
 7  spinoza                     -27.5517          0.0660592             44                      2                   42
 8  ss                         -100.135           0.0677818            150                      7                  143
 9  fleet                      -123.327           0.0702233            186                      9                  177
10  air_force                   -64.1203          0.0725086            100                      5                   95
11  armored                     -33.7666          0.0775565             56                      3                   53
12  wellington                  -30.2416          0.0833599             52                      3                   49
13  artillery                   -62.2033          0.084147             103                      6                   97
14  infantry                   -109.312           0.0876844            181                     11                  170
15  jeffersonian                -37.3645          0.0887561             65                      4                   61
16  kentuckians                 -26.7642          0.0901021             48                      3                   45
```

## Topics
So far we have looked at individual terms. What about topics?


```python
divergence_analysis(d, c1, c2, c1_name='female author', c2_name='male author',
                    topics_or_terms='topics', number_of_terms_to_print=20)
pass
```

                 female author    female author freq    male author    male author freq
    ---------  ---------------  --------------------  -------------  ------------------
    1976-1984              989             0.114348            2136           0.164497
    1985-1989              950             0.109839            1698           0.130766
    1990-1994             1183             0.136779            1854           0.14278
    1995-1999             1696             0.196092            2340           0.180208
    2000-2004             1762             0.203723            2323           0.178899
    2005-2009             1445             0.167071            1922           0.148017
    2010-2015              624             0.0721471            712           0.0548325
    
    
    Terms distinctive for Corpus 1: female author. 8649 Theses
    
        term                                      dunning    frequency_score    frequency_total    frequency female author    frequency male author
    --  -------------------------------------  ----------  -----------------  -----------------  -------------------------  -----------------------
    69  (28) Gender                            124041               0.925289         0.0180192                  0.040199                 0.0032458
    68  (48) Sexuality                          16681.1             0.771839         0.00734673                 0.0127279                0.00376246
    67  (35) Family/Household                   17031.3             0.768509         0.00771539                 0.013289                 0.00400293
    66  (36) Medical                            11380               0.697978         0.0101497                  0.0153898                0.00665933
    65  (22) Cultural - gender/class             9901.72            0.656113         0.0147091                  0.0205901                0.0107918
    64  (15) Art/ African American Art/Dance     4279.1             0.629933         0.00936309                 0.0124444                0.0073107
    63  (59) Film/Art                            2675.5             0.610286         0.00824304                 0.0105266                0.00672203
    62  (63) Cultural - Identity construction   11618               0.606814         0.0382537                  0.0485025                0.0314272
    61  (33) Education                           3515.12            0.587612         0.0174324                  0.0212326                0.0149011
    60  (19) Literary                            2261.66            0.587228         0.0113179                  0.013774                 0.00968196
    59  (11) Medieval                            1759.77            0.571895         0.0130949                  0.0154222                0.0115447
    58  (61) Cultural - symbolic myth            1350.96            0.568603         0.011064                   0.0129379                0.00981592
    57  (8) Rural                                1419.33            0.566135         0.0125276                  0.0145709                0.0111666
    56  (58) African American                    1342.81            0.560051         0.0144309                  0.0165628                0.013011
    55  (10) Civil rights                         531.384           0.547184         0.00932326                 0.0103998                0.00860622
    54  (70) Noise                                216.342           0.545266         0.00412831                 0.00458525               0.00382395
    53  (37) Jewish                               323.281           0.542644         0.00696267                 0.00768792               0.0064796
    52  (49) Labor                                740.618           0.542421         0.0161226                  0.017793                 0.0150099
    51  (4) Labor                                 143.692           0.539832         0.00355235                 0.00389758               0.0033224
    50  (45) Music                                339.891           0.538277         0.00910967                 0.00995988               0.00854336
    
    
    Terms distinctive for Corpus 2: male author. 12985 Theses
    
        term                                              dunning    frequency_score    frequency_total    frequency female author    frequency male author
    --  ---------------------------------------------  ----------  -----------------  -----------------  -------------------------  -----------------------
     0  (55) Military                                  -34723.5             0.250854         0.0244047                  0.0111322                0.0332451
     1  (6) US Political History; US Presidents         -9428.9             0.307512         0.0109975                  0.00627925               0.0141403
     2  (44) British History                           -13568.4             0.326586         0.0194086                  0.0118532                0.024441
     3  (67) International Relations/Latin America      -8259.06            0.347593         0.0152065                  0.00996269               0.0186993
     4  (40) Civil War                                  -7146.3             0.349406         0.0134693                  0.00887666               0.0165283
     5  (23) Economic/Business                          -2458.29            0.373608         0.00652769                 0.00464239               0.00778344
     6  (52) Central and Eastern Europe/ 20th Century   -5724.69            0.382886         0.0176469                  0.0129076                0.0208037
     7  (25) Political Parties                          -5664.52            0.39657          0.0222757                  0.0169644                0.0258134
     8  (42) Intellectual                               -6332.84            0.402813         0.0281392                  0.0218196                0.0323485
     9  (21) Roman empire                               -1266.81            0.403707         0.00573278                 0.00445671               0.00658275
    10  (12) Ottoman empire                             -1578.98            0.412275         0.00857996                 0.00683429               0.00974272
    11  (41) East Asia                                  -1476.16            0.417966         0.00915156                 0.00740653               0.0103139
    12  (54) Sports                                      -688.446           0.427123         0.00538812                 0.00447214               0.00599823
    13  (2) US Oil  / Regional (Oklahoma)                -306.173           0.438107         0.00330712                 0.00282759               0.00362652
    14  (50) Agrarian/Rural                             -1833.67            0.4393           0.0205743                  0.0176472                0.022524
    15  (46) History of the West                        -1304.52            0.44215          0.0160951                  0.0139104                0.0175504
    16  (34) Church                                     -1806.99            0.443326         0.0232165                  0.0201277                0.0252739
    17  (65) Biography ; US Presidents                  -1495.97            0.444581         0.0200897                  0.0174748                0.0218314
    18  (60) Italian Renaissance                         -442.82            0.451898         0.00786837                 0.00697688               0.00846218
    19  (47) Classics                                    -447.362           0.460536         0.0117613                  0.0106643                0.012492



The patterns are very similar. Gender, sexuality, and family/household are most distinctive for women; military, US political history, and british history 
are most distinctive for men. And again, no topic is nearly as distinctive for men as gender is for women. Gender has a frequency of 0.93 vs 0.25 for military.


## Gender vs. Sexuality
One thing that struck me as curious, though, was the difference between gender and sexuality. I tend to think of them as very closely related, e.g.
"gender and sexuality studies." The sexuality topic is still clearly female but less strongly than sexuality. 

Let's first have a look at the terms that make up the topics. 


```python
from topics import TOPICS

for topic_id in [28, 48]:
    print()
    print(TOPICS[topic_id]['name'])
    print('terms, prob', TOPICS[topic_id]['terms_prob'][:8])
    print('terms, frex', TOPICS[topic_id]['terms_frex'][:8])
    print('Highest scoring dissertations: ')
    d.print_dissertations_mentioning_terms_or_topics([f'topic.{topic_id}'])
```

    
    Gender
    terms, prob ['women', 'gender', 'femal', 'woman', 'men', 'work', 'feminist', 'live']
    terms, frex ['women', 'clubwomen', 'sisterhood', 'woman', 'suffragist', 'motherhood', 'ywca', 'femin']
    Highest scoring dissertations: 
    1989 Author: female  Advisor: unknown Day nurseries and wage-earning mothers in the United States, 1890-1930
    1988 Author: female  Advisor: female  "Women adrift" and "urban pioneers": Self-supporting working women in America, 1880-1930
    1991 Author: female  Advisor: male    Testing the boundaries: Women, politics, and gender roles in Chicago, 1890-1930
    2010 Author: female  Advisor: male    Jovita's legacy: Gender and women's agency in a south Texas family in the early twentieth century
    1997 Author: female  Advisor: unknown The Catholic woman's experience in nineteenth century America
    
    Sexuality
    terms, prob ['sexual', 'marriag', 'sex', 'prostitut', 'gender', 'moral', 'reproduct', 'men']
    terms, frex ['sexual', 'lesbian', 'homosexu', 'gay', 'heterosexu', 'birth_control', 'prostitut', 'same-sex']
    Highest scoring dissertations: 
    1981 Author: female  Advisor: unknown THE ICONOGRAPHY OF SAPPHO, 1775-1875
    1996 Author: female  Advisor: female  Sisters in sin: The image of the prostitute on the New York stage (1899-1918)
    1995 Author: female  Advisor: male    Gender differentiation and narrative construction in Propertius
    2002 Author: female  Advisor: male    Girl troubles: Female adolescent sexuality in the United States, 1850--1980
    2000 Author: female  Advisor: female  The politics of pleasure: Sexuality in radical movements for liberation and the women's liberation movement, 1968--1975


Maybe the gender topic is more of a womens' history topic?

I guess gender certainly seems to be distributed across a number of topics including 35 (Family / household) and 22 (Cultural - gender/class):


```python
for topic_id in [35, 22]:
    print()
    print(TOPICS[topic_id]['name'])
    print('terms, prob', TOPICS[topic_id]['terms_prob'][:8])
    print('terms, frex', TOPICS[topic_id]['terms_frex'][:8])
    print('Highest scoring dissertations: ')
    d.print_dissertations_mentioning_terms_or_topics([f'topic.{topic_id}'])
```

    
    Family/Household
    terms, prob ['famili', 'children', 'home', 'use', 'studi', 'signific', 'age', 'attitud']
    terms, frex ['sex-rol', 't-test', 'preschool', 'satisfact', 'questionnair', 'home_econom', 'pretest', 'two-par']
    Highest scoring dissertations: 
    1988 Author: female  Advisor: male    A comparison of saving behavior of the baby boom generation with that of a prior comparable age cohort
    1982 Author: female  Advisor: unknown DUAL-EARNER MARRIAGES: THE FAMILY SOCIAL ENVIRONMENT AND DYADIC ADJUSTMENT AMONG COUPLES WITH VARYING PATTERNS OF OCCUPATIONAL COMMITMENT
    1980 Author: female  Advisor: unknown EFFECT OF WORK PATTERN ON WOMEN'S SATISFACTION WITH RETIREMENT
    1981 Author: male    Advisor: unknown WIVES' LABOR FORCE INVOLVEMENT AND HUSBANDS' FAMILY WORK: A DUAL SPOUSAL PERSPECTIVE
    1980 Author: female  Advisor: unknown SOCIAL-PSYCHOLOGICAL PREDICTORS OF ATTITUDES TOWARD PREMARITAL SEXUAL PERMISSIVENESS AMONG COLLEGE WOMEN
    
    Cultural - gender/class
    terms, prob ['cultur', 'class', 'popular', 'ideal', 'gender', 'middle-class', 'imag', 'domest']
    terms, frex ['masculin', 'middle-class', 'manhood', 'beauti', 'leisur', 'victorian', 'genteel', 'childhood']
    Highest scoring dissertations: 
    2004 Author: female  Advisor: unknown Reflecting self-image: "Girlhood" interiors, 1875--1910
    1988 Author: female  Advisor: male    Culture and comfort: Parlor making in America, 1850-1930. (Volumes I and II)
    2005 Author: male    Advisor: female  Turning the tables: American restaurant culture and the rise of the middle class, 1880--1920
    1999 Author: female  Advisor: female  From Catharine Beecher to Martha Stewart: A cultural history of domestic advice
    2000 Author: female  Advisor: male    Self-made men: The margins of manliness among northern industrial workers, 1850--1920


Family/Household, even after excluding most home economics dissertations still remains home economics oriented, i.e. a mix of 
history of the family and the home on the one hand and social science (t-test, questionnaire, significance) on the other. 



## Most divergent topics over time
Quick detour: Let's look at the most divergent topics over time.

In the following plot, we can see the trajectories of the 5 most male and female topics from the 1980s to the 2010s. 
Line thickness indicates the overall weight of the topic.

Remember, a value of 0.95 as we find for much of the gender topic indicates that the topic has an almost 20 times higher weight
in female dissertations compared to male disserations.

I guess what's pretty bleak about the chart is that we can barely discern any changes. Gender / women's history starts out dominated
by female authors and ends almost in the same spot. Only Family/household becomes less gendered but I suspect that this is an artifact of 
home economics theses in the 1980s, which skew heavily female.


```python
from plot_gender_development import plot_gender_development_over_time
plot_gender_development_over_time(
    no_terms_or_topics_to_show=10,
    data='topics',
    display_selector='most_divergent',
    store_to_filename='divergent_topics.png',
    show_plot=False)
pass
```

![alt text](https://github.com/srisi/gender_history/raw/master/data/divergent_topics.png)

What about individual terms? Let's look at the most frequent terms in the gender topic:

The only major change seems to be for gender, which starts out with a frequency score of about 0.9 and ends at about 0.8, i.e. it moves from a
9:1 female/male ratio to a 4:1 ratio. Not super encouraging.


```python
plot_gender_development_over_time(
    no_terms_or_topics_to_show=8,
    data='terms',
    selected_terms_or_topics=['women', 'gender', 'female', 'woman', 'men', 'work', 'feminist'],
    store_to_filename='frequent_gender_terms.png',
    title='Most frequent gender and womens\' history terms',
    show_plot=False)
pass
```

![alt text](https://github.com/srisi/gender_history/raw/master/data/frequent_gender_terms.png)

## Sexuality, Male vs. Female Authors
Anway, let's move back to the difference between the gender/women's history topic and the sexuality topic. Why is the sexuality topic
less distinctive for female authors?

Let's only retain dissertations that score in the top 20% for the sexuality topic and compare male to female authors:


```python
d = Dataset()

# Retain only dissertations that score in the top 20% for topic 48 (sexuality)
d.topic_percentile_score_filter(topic=48, min_percentile_score=80)

# Create two sub-datasets, one for female authors and one for male authors
c1 = d.copy().filter(author_gender='female')
c2 = d.copy().filter(author_gender='male')

# Run the divergence analysis
divergence_analysis(d, c1, c2, c1_name='female author', c2_name='male author',
                    topics_or_terms='terms', number_of_terms_to_print=10)
pass
```

                 female author    female author freq    male author    male author freq
    ---------  ---------------  --------------------  -------------  ------------------
    1976-1984              204             0.0659341            148           0.120032
    1985-1989              243             0.0785391             98           0.0794809
    1990-1994              401             0.129606             151           0.122466
    1995-1999              693             0.223982             233           0.18897
    2000-2004              751             0.242728             307           0.248986
    2005-2009              567             0.183258             210           0.170316
    2010-2015              235             0.0759535             86           0.0697486
    
    
    Terms distinctive for Corpus 1: female author. 3094 Theses
    
          term         dunning    frequency_score    count_total    count female author    count male author
    ----  ---------  ---------  -----------------  -------------  ---------------------  -------------------
    2324  woman      2310.86             0.778415          11529                  10349                 1180
    2323  female      254.589            0.745843           1607                   1414                  193
    2322  women       239.306            0.773046           1245                   1114                  131
    2321  feminist    234.95             0.829032            865                    799                   66
    2320  gender      205.489            0.665203           2712                   2257                  455
    2319  feminism    109.598            0.81923             431                    396                   35
    2318  family       76.5753           0.601897           2521                   1993                  528
    2317  womanhood    76.4478           0.889024            210                    200                   10
    2316  suffrage     67.4622           0.807608            287                    262                   25
    2315  gendered     64.7725           0.742325            426                    374                   52
    
    
    Terms distinctive for Corpus 2: male author. 1233 Theses
    
        term             dunning    frequency_score    count_total    count female author    count male author
    --  -------------  ---------  -----------------  -------------  ---------------------  -------------------
     0  gay            -179.562           0.128387             212                     57                  155
     1  homosexual     -102.513           0.13143              124                     34                   90
     2  crime           -95.6376          0.277502             429                    210                  219
     3  criminal        -66.7774          0.285993             328                    164                  164
     4  homosexuality   -60.0944          0.185727             113                     41                   72
     5  masculinity     -55.7411          0.294232             300                    153                  147
     6  gambling        -54.031           0.0992211             51                     11                   40
     7  juvenile        -48.2988          0.197107             100                     38                   62
     8  city            -48.2018          0.406227            1484                    936                  548
     9  slave           -44.8023          0.378072             783                    472                  311



This is quite striking. When men write about sexuality, the most distinctive terms relate to male homosexuality.

Here's an intriguing idea: Can we in any way make a case that if (mostly) female historians hadn't established gender history in the first place, 
queer history wouldn't have taken off in the same way? Or do gay history and womens' history have completely separate trajectories that accidentally got
bundeled together in the topic model? Let's look at the dissertations that score highly for gay history:


Sidenote: the sexuality topic is at least in part also an urban vice topic (crime, gambling, city, juvenile...) For right now, that's not our focus.




```python
# get the dissertations that mention gay, homosexual, homosexuality, masculinity most frequently
d.print_dissertations_mentioning_terms_or_topics(terms=['gay', 'homosexual', 'homosexuality', 'masculinity'], no_dissertations=30)
```

    2014 Author: male    Advisor: male    City, Suburb, and the Changing Bounds of Lesbian and Gay Life and Politics in Metropolitan Detroit, 1945-1985
    2005 Author: male    Advisor: female  American homophobia: "The homosexual menace" in twentieth-century American culture
    2006 Author: female  Advisor: female  Persistent pathologies: The odd coupling of alcoholism and homosexuality in the discourses of twentieth century science
    1982 Author: male    Advisor: unknown OUT OF THE SHADOWS: THE GAY EMANCIPATION MOVEMENT IN THE UNITED STATES, 1940-1970
    2008 Author: male    Advisor: unknown Lavender sons of Zion: A history of gay men in Salt Lake City, 1950--1979
    2009 Author: female  Advisor: unknown The unapologetic athlete: The Gay Games, 1982-1994
    2014 Author: male    Advisor: male    Special Relationships: Transnational Homophile Activism and Anglo-American Sexual Politics
    1998 Author: male    Advisor: male    Americans' attitudes toward gays and lesbians since 1965
    2013 Author: male    Advisor: female  No Place Like Home: A Cultural History of Gay Domesticity, 1948???1982
    2005 Author: male    Advisor: female  The company he keeps: White college fraternities, masculinity, and power, 1825--1975
    2000 Author: female  Advisor: female  Cape Queer: The politics of sex, class, and race in Provincetown, Massachusetts, 1859--1999
    2006 Author: male    Advisor: male    Passionate anxieties: McCarthyism and homosexual identities in the United States, 1945--1965
    1990 Author: male    Advisor: male    Male homosexuality and its regulation in late medieval Florence. (Volumes I and II)
    2006 Author: male    Advisor: female  Arrested development: Homosexuality, gender, and American adolescence, 1890--1930
    1996 Author: male    Advisor: male    Beyond Carnival: Homosexuality in twentieth-century Brazil
    2007 Author: male    Advisor: male    Middle-class masculinity and the Klondike gold rush
    2002 Author: female  Advisor: female  Inverts, perverts, and national peril: Federal responses to homosexuality, 1890--1956
    2004 Author: female  Advisor: unknown Playing the man: Masculinity, performance, and United States foreign policy, 1901--1920
    2007 Author: male    Advisor: female  Radical relations: A history of lesbian and gay parents and their children in the United States, 1945--2003
    2005 Author: male    Advisor: male    Working out Egypt: Masculinity and subject formation between colonial modernity and nationalism, 1870--1940
    2005 Author: male    Advisor: unknown The streets of San Francisco: Blacks, beats, homosexuals, and the San Francisco Police Department, 1950--1968
    2006 Author: male    Advisor: male    "A part of our liberation": ONE Magazine and the cultivation of gay liberation, 1953--1963
    2013 Author: male    Advisor: unknown The Golden Age of Gay Nightlife: Performing Glamour and Deviance in Los Angeles and West Hollywood, 1966???2013
    2002 Author: male    Advisor: male    The Nazi "new man": Embodying masculinity and regulating sexuality in the SA and SS, 1930--1939
    2004 Author: male    Advisor: female  (Un)making macho: Race, gender, and stardom in 1970s American cinema and culture
    1997 Author: male    Advisor: male    Becoming a man in Kwawu: Gender, law, personhood, and the construction of masculinities in colonial Ghana, 1875-1957
    1999 Author: male    Advisor: male    "Eradicating this menace": Homophobia and anti-communism in Congress, 1947-1954
    2011 Author: male    Advisor: male    Urban Desires: Practicing Pleasure in the ?? City of Light,??? 1848???1900
    1989 Author: male    Advisor: unknown Gay New York: Urban culture and the making of a gay male world, 1890-1940
    1995 Author: female  Advisor: female  San Francisco was a wide open town: Charting the emergence of lesbian and gay communities through the mid-twentieth century


The fact that there are many more women among the advisors than the advisees is certainly suggestive

At any rate, if we look at the gender topic as more of a women's history topic and see the strong queer history focus in the sexuality topic, it makes sense
that the sexuality topic is less gendered than the women's history topic. Keep in mind, though, that sexuality is the second most gendered topic by frequency score 
(0.77) after gender (0.93)

## Advisor Gender
We can also look at the gender of the advisor and try to assess their effects. 

If you're a man with a female advisor, what are terms and topics that you are more likely to write about?

One problem with this comparison is that the combination male student, female advisor appears later in our dataset than
male student, male advisor because faculty jobs remained dominated by men. 
As a result if we compare these datasets from 1976 to 2015, we'll get many effects that are primarily the product of changing historiography overall,
e.g. the replacement of social with cultural history.
```
             female advisor    female advisor freq    male advisor    male advisor freq
---------  ----------------  ---------------------  --------------  -------------------
1976-1984                 0              0                      10           0.00152718
1985-1989                35              0.0275591             412           0.06292
1990-1994               143              0.112598             1183           0.180666
1995-1999               234              0.184252             1575           0.240531
2000-2004               364              0.286614             1688           0.257789
2005-2009               342              0.269291             1230           0.187844
2010-2015               152              0.119685              450           0.0687233
```

What we'll do instead is limit this analysis to the years 2000-2015 for which the datasets are more balanced:
```
             female advisor    female advisor freq    male advisor    male advisor freq
---------  ----------------  ---------------------  --------------  -------------------
2000-2004               364               0.424242            1688             0.501188
2005-2009               342               0.398601            1230             0.365202
2010-2015               152               0.177156             450             0.13361
```



```python
d = Dataset()

# retain only dissertations written by men between 2000 and 2015
d.filter(author_gender='male', start_year=2000, end_year=2015)

# Create two sub-datasets, one for female advisors and one for male advisors
c1 = d.copy().filter(advisor_gender='female')
c2 = d.copy().filter(advisor_gender='male')

# Run the divergence analysis

divergence_analysis(d, c1, c2, c1_name='female advisor', c2_name='male advisor',
                    topics_or_terms='topics', number_of_terms_to_print=6)
divergence_analysis(d, c1, c2, c1_name='female advisor', c2_name='male advisor',
                    topics_or_terms='terms', number_of_terms_to_print=12)
pass
```

                 female advisor    female advisor freq    male advisor    male advisor freq
    ---------  ----------------  ---------------------  --------------  -------------------
    2000-2004               364               0.424242            1688             0.501188
    2005-2009               342               0.398601            1230             0.365202
    2010-2015               152               0.177156             450             0.13361
    
    
    Terms distinctive for Corpus 1: female advisor. 858 Theses
    
        term                                    dunning    frequency_score    frequency_total    frequency female advisor    frequency male advisor
    --  ------------------------------------  ---------  -----------------  -----------------  --------------------------  ------------------------
    69  (48) Sexuality                          842.473           0.695137         0.00462235                  0.00836544                0.0036688
    68  (28) Gender                             371.291           0.6487           0.00388734                  0.00612542                0.00331719
    67  (11) Medieval                           848.674           0.633622         0.0113665                   0.0171219                 0.00990035
    66  (59) Film/Art                           641.432           0.627547         0.00954483                  0.0141188                 0.0083796
    65  (15) Art/ African American Art/Dance    468.765           0.613864         0.00899419                  0.0127696                 0.0080324
    64  (22) Cultural - gender/class            512.926           0.594668         0.0147804                   0.0198062                 0.0135001
    
    
    Terms distinctive for Corpus 2: male advisor. 3368 Theses
    
        term                                       dunning    frequency_score    frequency_total    frequency female advisor    frequency male advisor
    --  ---------------------------------------  ---------  -----------------  -----------------  --------------------------  ------------------------
     0  (12) Ottoman empire                       -803.171           0.295314         0.00762462                  0.00362252                0.00864416
     1  (55) Military                            -2871.96            0.318238         0.0337147                   0.0176481                 0.0378076
     2  (16) Islamic                              -284.036           0.368694         0.00604322                  0.00385492                0.00660069
     3  (26) Labor                                -529.924           0.369386         0.0113715                   0.00727256                0.0124157
     4  (40) Civil War                            -453.897           0.403892         0.017219                    0.012484                  0.0184252
     5  (6) US Political History; US Presidents   -343.636           0.405566         0.0134772                   0.00982918                0.0144065
                 female advisor    female advisor freq    male advisor    male advisor freq
    ---------  ----------------  ---------------------  --------------  -------------------
    2000-2004               364               0.424242            1688             0.501188
    2005-2009               342               0.398601            1230             0.365202
    2010-2015               152               0.177156             450             0.13361
    
    
    Terms distinctive for Corpus 1: female advisor. 858 Theses
    
          term           dunning    frequency_score    count_total    count female advisor    count male advisor
    ----  -----------  ---------  -----------------  -------------  ----------------------  --------------------
    2382  woman          65.8805           0.66893             641                     219                   422
    2381  black          65.488            0.619401           1418                     418                  1000
    2380  gender         46.5763           0.68348             370                     132                   238
    2379  female         38.5998           0.78232             100                      48                    52
    2378  identity       38.5656           0.598598           1271                     352                   919
    2377  michigan       36.5697           0.847607             51                      30                    21
    2376  race           34.0112           0.61408             810                     235                   575
    2375  film           33.3462           0.659719            367                     122                   245
    2374  african        31.4128           0.594673           1129                     309                   820
    2373  activism       30.3071           0.692262            213                      78                   135
    2372  masculinity    27.8056           0.720667            138                      55                    83
    2371  art            27.1231           0.683528            213                      76                   137
    
    
    Terms distinctive for Corpus 2: male advisor. 3368 Theses
    
        term          dunning    frequency_score    count_total    count female advisor    count male advisor
    --  ----------  ---------  -----------------  -------------  ----------------------  --------------------
     0  irish        -65.0823          0.0918397            237                       6                   231
     1  army         -62.2199          0.277861             656                      59                   597
     2  war          -58.9727          0.413529            3463                     531                  2932
     3  british      -38.9519          0.352825             863                     106                   757
     4  al           -36.5928          0.0609358            122                       2                   120
     5  military     -34.5574          0.377376            1069                     144                   925
     6  imperial     -33.6795          0.305054             454                      46                   408
     7  germany      -30.397           0.319973             473                      51                   422
     8  combat       -29.6144          0.203381             195                      12                   183
     9  cavalry      -27.6804          0                     76                       0                    76
    10  eisenhower   -27.1845          0.0992536            109                       3                   106
    11  failure      -26.2126          0.265469             259                      22                   237


I think this is quite interesting. 

A male student with a female advisor is far more likely to write about sexuality and gender. The same goes for race. In each case, the ratio is about 1:2 male to female
advisor. (I don't know why medieval scores so highly here. Is there a reason why medieval history has a lot of female faculty?)



```python
d = Dataset()

# retain only dissertations written by men between 2000 and 2015
d.filter(author_gender='female', start_year=2000, end_year=2015)

# Create two sub-datasets, one for female advisors and one for male advisors
c1 = d.copy().filter(advisor_gender='female')
c2 = d.copy().filter(advisor_gender='male')

# Run the divergence analysis

divergence_analysis(d, c1, c2, c1_name='female advisor', c2_name='male advisor',
                    topics_or_terms='topics', number_of_terms_to_print=5)
divergence_analysis(d, c1, c2, c1_name='female advisor', c2_name='male advisor',
                    topics_or_terms='terms', number_of_terms_to_print=10)
pass
```

                 female advisor    female advisor freq    male advisor    male advisor freq
    ---------  ----------------  ---------------------  --------------  -------------------
    2000-2004               588               0.446809             971             0.492144
    2005-2009               489               0.371581             703             0.35631
    2010-2015               239               0.181611             299             0.151546
    
    
    Terms distinctive for Corpus 1: female advisor. 1316 Theses
    
        term                                        dunning    frequency_score    frequency_total    frequency female advisor    frequency male advisor
    --  ----------------------------------------  ---------  -----------------  -----------------  --------------------------  ------------------------
    69  (28) Gender                                3162.93            0.642931         0.0372508                   0.0508001                 0.0282133
    68  (48) Sexuality                             1202.87            0.636758         0.0155466                   0.0209431                 0.0119471
    67  (35) Family/Household                       329.107           0.600109         0.00814991                  0.0101892                 0.0067897
    66  (22) Cultural - gender/class                837.829           0.591616         0.0249215                   0.0306082                 0.0211284
    65  (69) Noise; instructions for filing diss    106.453           0.589527         0.00331845                  0.00405777                0.00282533
    
    
    Terms distinctive for Corpus 2: male advisor. 1973 Theses
    
        term                        dunning    frequency_score    frequency_total    frequency female advisor    frequency male advisor
    --  ------------------------  ---------  -----------------  -----------------  --------------------------  ------------------------
     0  (23) Economic/Business     -557.05            0.290525         0.00362349                  0.00194283                0.0047445
     1  (47) Classics             -1103.31            0.314709         0.00911861                  0.00534382                0.0116364
     2  (60) Italian Renaissance   -371.154           0.36261          0.00550819                  0.0037868                 0.00665637
     3  (9) Latin America          -688.797           0.363815         0.010395                    0.00717345                0.0125439
     4  (40) Civil War             -552.25            0.373154         0.00957802                  0.00680339                0.0114287
                 female advisor    female advisor freq    male advisor    male advisor freq
    ---------  ----------------  ---------------------  --------------  -------------------
    2000-2004               588               0.446809             971             0.492144
    2005-2009               489               0.371581             703             0.35631
    2010-2015               239               0.181611             299             0.151546
    
    
    Terms distinctive for Corpus 1: female advisor. 1316 Theses
    
          term         dunning    frequency_score    count_total    count female advisor    count male advisor
    ----  ---------  ---------  -----------------  -------------  ----------------------  --------------------
    1889  woman       442.148            0.65007            4636                    2595                  2041
    1888  feminist    122.619            0.76286             381                     262                   119
    1887  gender       98.8354           0.640959           1182                     650                   532
    1886  women        62.5609           0.66927             508                     295                   213
    1885  dress        53.4788           0.930759             51                      46                     5
    1884  activism     52.6076           0.694578            317                     193                   124
    1883  sexual       52.3606           0.685657            349                     209                   140
    1882  lesbian      49.0499           0.899491             57                      49                     8
    1881  sexuality    48.4592           0.725989            211                     136                    75
    1880  fashion      48.2502           0.766322            146                     101                    45
    
    
    Terms distinctive for Corpus 2: male advisor. 1973 Theses
    
        term           dunning    frequency_score    count_total    count female advisor    count male advisor
    --  -----------  ---------  -----------------  -------------  ----------------------  --------------------
     0  jewish        -49.8353          0.329368             477                     120                   357
     1  polish        -38.9807          0.086145              66                       4                    62
     2  poland        -34.561           0.0929989             61                       4                    57
     3  indian        -33.9517          0.328431             323                      81                   242
     4  spanish       -33.8803          0.319454             292                      71                   221
     5  bank          -31.4669          0.113607              62                       5                    57
     6  britain       -28.2443          0.308537             218                      51                   167
     7  reformation   -26.6715          0.202081              88                      13                    75
     8  imperial      -25.2359          0.368865             406                     116                   290
     9  british       -25.1774          0.376057             452                     132                   320


The same holds true for female students with male vs. female advisors. I guess this is less surprising.


## Microgenres

One thing that strikes me is that when men and women write about one topic, they often do so in quite different ways.

Let's look at the science topic (24):


```python
d = Dataset()
# only include dissertations that score in the 80th percentile or above for the science topic.
d.topic_percentile_score_filter(topic=24, min_percentile_score=80)

# Create two sub-datasets, one for female authors and one for male authors
c1 = d.copy().filter(author_gender='female')
c2 = d.copy().filter(author_gender='male')

divergence_analysis(d, c1, c2, topics_or_terms='topics', c1_name='female', c2_name='male', 
                    sort_by='dunning', number_of_terms_to_print=5)
pass
```

                 female    female freq    male    male freq
    ---------  --------  -------------  ------  -----------
    1976-1984       226      0.133807      407    0.154284
    1985-1989       188      0.111308      330    0.125095
    1990-1994       195      0.115453      376    0.142532
    1995-1999       313      0.185317      445    0.168688
    2000-2004       344      0.203671      517    0.195982
    2005-2009       294      0.174067      396    0.150114
    2010-2015       129      0.0763766     167    0.0633055
    
    
    Terms distinctive for Corpus 1: female. 1689 Theses
    
        term                            dunning    frequency_score    frequency_total    frequency female    frequency male
    --  ----------------------------  ---------  -----------------  -----------------  ------------------  ----------------
    69  (28) Gender                    14931.5            0.92662          0.0105543            0.0240624        0.00190554
    68  (36) Medical                    6240.64           0.688611         0.0308498            0.0463192        0.0209454
    67  (48) Sexuality                  3121.35           0.768547         0.00701861           0.0122288        0.00368277
    66  (35) Family/Household           2855.53           0.77587          0.0060344            0.0106529        0.00307736
    65  (22) Cultural - gender/class    1786.47           0.619363         0.023385             0.0305681        0.0187859
    
    
    Terms distinctive for Corpus 2: male. 2638 Theses
    
        term                                             dunning    frequency_score    frequency_total    frequency female    frequency male
    --  ---------------------------------------------  ---------  -----------------  -----------------  ------------------  ----------------
     0  (42) Intellectual                              -3868.36            0.400408         0.0830002           0.0636859         0.0953664
     1  (55) Military                                  -3298.69            0.278574         0.0148977           0.00756544        0.0195923
     2  (6) US Political History; US Presidents        -1273.01            0.308206         0.00761048          0.00432715        0.00971265
     3  (52) Central and Eastern Europe/ 20th Century   -788.79            0.398083         0.0161812           0.0123316         0.0186459
     4  (34) Church                                     -607.245           0.413631         0.0172285           0.0137322         0.019467


Among women, it's the gender, medical, and sexuality topics predominate. Among men, intellectual, military, and US political history.

Gender, sexuality, and military aren't super interesting. They already skew male and female in the general dataset. However, 
the contrast between medical and intellectual is interesting. Men are more likely to write at the intersection of intellectual history and
history of science; women between medical history and history of science. 
Here are some examples: 


```python
# History of science, male-dominated areas
d.print_dissertations_mentioning_terms_or_topics(['topic.36', 'topic.28', 'topic.48'], no_dissertations=20)
```

    1999 Author: female  Advisor: male    Puerperal insanity: Women, psychiatry, and the asylum in Victorian England, 1820--1895
    2002 Author: female  Advisor: female  Rethinking the rise of scientific medicine: Trier, Germany, 1880--1914
    1999 Author: female  Advisor: female  'It did not seem like a hospital it seemed like home': Women's experiences as patients at Peterson's Hospital, Ann Arbor, Michigan, 1902-1933
    1997 Author: female  Advisor: male    Earnestly working to improve Russia's future: Russian women physicians, 1867-1905
    2000 Author: male    Advisor: female  "Of physick and astronomy": Almanacs and popular medicine in Massachusetts, 1700--1764
    1995 Author: female  Advisor: female  Doubtful sex: Cases and concepts of hermaphroditism in France and Britain, 1868-1915
    1999 Author: female  Advisor: male    A vital force: Women physicians and patients in American homeopathy, 1850--1930
    2006 Author: female  Advisor: male    'The wife your husband needs': Marriage counseling, religion, and sexual politics in the United States, 1930--1980
    1995 Author: female  Advisor: female  Hospital waifs: The hospital care of children in Boston, 1860-1920
    1993 Author: female  Advisor: male    Tuberculosis as chronic illness in the United States: Understanding, treating, and living with the disease, 1884-1954
    1995 Author: male    Advisor: male    The hemocytometer and its impact on Progressive Era medicine
    1986 Author: male    Advisor: unknown SCIENTIFIC MEDICINE COMES TO PHILADELPHIA: PUBLIC HEALTH TRANSFORMED, 1854-1899 (BACTERIOLOGY, PENNSYLVANIA)
    1998 Author: female  Advisor: unknown Representing hysterectomy and its consequences, 1940's-1990's
    1995 Author: female  Advisor: male    "To relieve distressed women:" Teaching and establishing the scientific art of man-midwifery or gynecology in Edinburgh and London, 1720-1805
    2001 Author: female  Advisor: female  J. B. van Helmont's heuristic wound: Trauma and the subversion of humoral theory
    1986 Author: male    Advisor: unknown MADNESS AND MEDICINE: THE MEDICAL APPROACH TO MADNESS IN ANTE-BELLUM AMERICA, WITH PARTICULAR REFERENCE TO THE EASTERN LUNATIC ASYLUM OF VIRGINIA AND THE SOUTH CAROLINA LUNATIC ASYLUM
    2001 Author: male    Advisor: female  Healing bodies and saving the race: Women, public health, eugenics, and sexuality, 1890--1950
    2010 Author: female  Advisor: female  Down and out in old J.D.: Urban public hospitals, institutional stigma and medical indigence in the twentieth century
    1994 Author: female  Advisor: female  Dirty discourse: Birth control advertising in the 1920s and 1930s
    1992 Author: female  Advisor: male    A social analysis of insanity in nineteenth-century Germany: Sexuality, delinquency, and anti-Semitism in the records of the Eberbach Asylum



```python
# History of science, female-dominated areas
d.print_dissertations_mentioning_terms_or_topics(['topic.42', 'topic.55', 'topic.6'], no_dissertations=20)
```

    1998 Author: male    Advisor: male    G. W. Leibniz: Personhood, moral agency, and meaningful immortality
    1994 Author: male    Advisor: male    Aquinas on gravitational motion: An investigation
    1990 Author: male    Advisor: unknown Seascape with fog: Analogy, certainty and cultural exemplars in John Locke's "An Essay Concerning Human Understanding"
    2015 Author: male    Advisor: male    Spinoza and the problem of universals: A study and research guide
    2006 Author: male    Advisor: unknown Acquiring "feelings that do not err": Moral deliberation and the sympathetic point of view in the ethics of Dai Zhen
    1997 Author: male    Advisor: unknown Jiddu Krishnamurti and Thich Nhat Hanh on the silence of God and the human condition
    2004 Author: male    Advisor: male    Men on "iron ponies", the death and rebirth of the modern United States cavalry
    1996 Author: male    Advisor: unknown Self, sympathy, and society in Hume's "Treatise of Human Nature"
    2003 Author: male    Advisor: male    The AEF way of war: The American Army and combat in the First World War
    1985 Author: male    Advisor: unknown THE SEARCH FOR AN AMERICAN STRATEGY: THE ORIGINS OF THE KENNEDY DOCTRINE, 1936 - 1961 (JOINT CHIEFS OF STAFF)
    1988 Author: male    Advisor: male    A sense of wonder: Reassessing the life and work of Ludwig Wittgenstein
    1989 Author: male    Advisor: unknown The 'prima via': Natural philosophy's approach to God
    1997 Author: male    Advisor: male    Descartes's metaphysical reasoning
    2012 Author: male    Advisor: male    David Hume's Account of Demonstration in Book I of ??úA Treatise of Human Nature??Ñ
    1996 Author: male    Advisor: male    The American soldier in Vietnam
    2001 Author: female  Advisor: unknown A fine group of fellows: Civilian advisors, Eisenhower, and national security planning
    2002 Author: male    Advisor: female  Rethinking rational cosmology: Research on the pre-critical origins of Kant's arguments in the antinomies
    1987 Author: male    Advisor: unknown THE ORIGINS OF MODERN HISTORICAL CONSCIOUSNESS, 1822-1848: HEGEL'S PHILOSOPHY OF HISTORY AND ITS CRITIQUE BY RANKE AND MARX
    1985 Author: male    Advisor: unknown THE MARXIAN TRANSCENDENCE OF PHILOSOPHY AND PHILOSOPHICAL MARXISM. (GERMAN TEXT)
    1989 Author: female  Advisor: unknown The Enlightenment legacy of David Hume


The point with the microgenres is: Even if two dissertations share the same overarching area (history of science; 19th century U.S. history etc.), 
we can identify substantial differences between the work of men and women. And I think this applies to all areas we might look at.

## Institutions
Just for fun, we can also apply the same analysis to institutions and compare, for example, Harvard dissertations against non-Harvard dissertations. 


```python
d = Dataset()
# only include dissertations that score in the 80th percentile or above for the science topic.

# Create two sub-datasets, one for harvard dissertations and one for non-harvard dissertations
c1 = d.copy().filter(institution_filter='harvard')
c2 = d.copy().filter(institution_filter='not_harvard')

divergence_analysis(d, c1, c2, topics_or_terms='topics', c1_name='harvard', c2_name='not harvard', 
                    sort_by='dunning', number_of_terms_to_print=8)
pass
```

                 harvard    harvard freq    not harvard    not harvard freq
    ---------  ---------  --------------  -------------  ------------------
    1976-1984         91       0.116967            3034           0.145474
    1985-1989        116       0.1491              2532           0.121404
    1990-1994        131       0.16838             2906           0.139336
    1995-1999        144       0.18509             3892           0.186613
    2000-2004         97       0.124679            3988           0.191216
    2005-2009        136       0.174807            3231           0.154919
    2010-2015         63       0.0809769           1273           0.0610376
    
    
    Terms distinctive for Corpus 1: harvard. 778 Theses
    
        term                   dunning    frequency_score    frequency_total    frequency harvard    frequency not harvard
    --  -------------------  ---------  -----------------  -----------------  -------------------  -----------------------
    69  (47) Classics          4515.66           0.724869         0.0117613             0.0292664               0.0111084
    68  (41) East Asia         4498.59           0.743614         0.00915156            0.024845                0.00856614
    67  (16) Islamic           2877.1            0.738249         0.00627595            0.0166132               0.00589033
    66  (12) Ottoman empire    1765.85           0.680595         0.00857996            0.017568                0.00824468
    65  (66) Political         1425.63           0.624211         0.0179511             0.0291256               0.0175342
    64  (64) Christian         1376.09           0.625349         0.016949              0.0276257               0.0165507
    63  (24) Science           1318.72           0.633364         0.0139637             0.0235074               0.0136077
    62  (62) GENERIC           1004.23           0.588153         0.028163              0.0396094               0.027736
    
    
    Terms distinctive for Corpus 2: not harvard. 20856 Theses
    
        term                              dunning    frequency_score    frequency_total    frequency harvard    frequency not harvard
    --  ------------------------------  ---------  -----------------  -----------------  -------------------  -----------------------
     0  (55) Military                   -2575.82            0.288886         0.0244047            0.0101306                0.0249372
     1  (40) Civil War                  -1764.1             0.259566         0.0134693            0.0048347                0.0137914
     2  (39) History of the West        -1716.96            0.261939         0.0133265            0.00484194               0.013643
     3  (18) Native american            -1392.5             0.241171         0.00941868           0.00306873               0.00965556
     4  (28) Gender                     -1216.82            0.33774          0.0180192            0.00935431               0.0183424
     5  (65) Biography ; US Presidents  -1166.03            0.35137          0.0200897            0.0110652                0.0204264
     6  (54) Sports                      -927.749           0.215804         0.00538812           0.00152245               0.00553232
     7  (56) Civil Rights                -669.517           0.368645         0.0143184            0.00848743               0.0145359


Not super useful but I'm sure Londa will get a kick out of seeing that gender is one of the most under-represented topics among 
Harvard dissertations. 

## Descendants vs. No Descendants
Are there substantial differences between historians who advise other historians and those who do not?

Note: the following analysis is limited to the years between 1980 and 2004 because few historians who graduated
after 2004 have any descendants yet, i.e. the "no descendants" dataset would be skewed towards the present without
2004 as an end year.


```python
d = Dataset()

# retain only dissertations written by men between 2000 and 2015
d.filter(start_year=1980, end_year=2004)

# Create two sub-datasets, one for female advisors and one for male advisors
c1 = d.copy().filter(has_descendants=True)
c2 = d.copy().filter(has_descendants=False)

# Run the divergence analysis

divergence_analysis(d, c1, c2, c1_name='has descendants', c2_name='no descendants',
                    topics_or_terms='topics', number_of_terms_to_print=6, 
                   sort_by='dunning')
pass

```

                 has descendants    has descendants freq    no descendants    no descendants freq
    ---------  -----------------  ----------------------  ----------------  ---------------------
    1976-1984                354                0.233972              2685               0.175124
    1985-1989                333                0.220093              2315               0.150991
    1990-1994                288                0.19035               2749               0.179298
    1995-1999                331                0.218771              3705               0.241651
    2000-2004                207                0.136814              3878               0.252935
    
    
    Terms distinctive for Corpus 1: has descendants. 1513 Theses
    
        term                                     dunning    frequency_score    frequency_total    frequency has descendants    frequency no descendants
    --  -------------------------------------  ---------  -----------------  -----------------  ---------------------------  --------------------------
    69  (26) Labor                              2200.27            0.62036          0.0169979                     0.0262792                  0.016082
    68  (41) East Asia                          1379.18            0.62844          0.00913837                    0.0145526                  0.00860409
    67  (63) Cultural - Identity construction    934.41            0.565053         0.0286603                     0.0362593                  0.0279104
    66  (58) African American                    871.293           0.588498         0.0135883                     0.0187101                  0.0130829
    65  (43) Development                         678.244           0.544466         0.0468407                     0.0550204                  0.0460335
    64  (46) History of the West                 464.966           0.562765         0.0154019                     0.0193254                  0.0150147
    
    
    Terms distinctive for Corpus 2: no descendants. 15332 Theses
    
        term                                       dunning    frequency_score    frequency_total    frequency has descendants    frequency no descendants
    --  ---------------------------------------  ---------  -----------------  -----------------  ---------------------------  --------------------------
     0  (65) Biography ; US Presidents           -2251.93            0.355296          0.0213874                   0.0122818                    0.022286
     1  (55) Military                            -1441.25            0.394982          0.0243101                   0.0163814                    0.0250925
     2  (33) Education                           -1384.05            0.381242          0.0186967                   0.0119311                    0.0193643
     3  (44) British History                     -1318.26            0.391877          0.0210925                   0.0140405                    0.0217884
     4  (6) US Political History; US Presidents  -1314.78            0.347536          0.0113929                   0.00633434                   0.0118921
     5  (40) Civil War                            -805.659           0.393762          0.013313                    0.00892806                   0.0137457


Actually, that's quite intriguing and somewhat encouraging: if you work on gender or non-western topics, you're more
likely to have descendants (and presumably get a tenure-track job). Conversely, if you work on US presidents or military
history, you're less likely to get a job. 

The individual terms, see below, are less useful.


```python
divergence_analysis(d, c1, c2, c1_name='has descendants', c2_name='no descendants',
                    topics_or_terms='terms', number_of_terms_to_print=12,
                    sort_by='dunning')
pass
```

                 has descendants    has descendants freq    no descendants    no descendants freq
    ---------  -----------------  ----------------------  ----------------  ---------------------
    1976-1984                354                0.233972              2685               0.175124
    1985-1989                333                0.220093              2315               0.150991
    1990-1994                288                0.19035               2749               0.179298
    1995-1999                331                0.218771              3705               0.241651
    2000-2004                207                0.136814              3878               0.252935
    
    
    Terms distinctive for Corpus 1: has descendants. 1513 Theses
    
          term        dunning    frequency_score    count_total    count has descendants    count no descendants
    ----  --------  ---------  -----------------  -------------  -----------------------  ----------------------
    6653  social     132.865            0.589759           9638                     1264                    8374
    6652  labor      111.674            0.631025           3383                      515                    2868
    6651  worker      90.0283           0.633656           2596                      399                    2197
    6650  race        80.7562           0.638501           2135                      334                    1801
    6649  class       65.9567           0.587999           4980                      649                    4331
    6648  black       56.8999           0.580862           5180                      658                    4522
    6647  colonial    53.2412           0.6064             2607                      363                    2244
    6646  maya        46.8018           0.865387             67                       27                      40
    6645  politics    46.7843           0.581015           4233                      538                    3695
    6644  cultural    43.6059           0.576128           4524                      565                    3959
    6643  douglass    42.8274           0.851067             72                       27                      45
    6642  harlem      42.4472           0.82877              92                       31                      61
    
    
    Terms distinctive for Corpus 2: no descendants. 15332 Theses
    
        term           dunning    frequency_score    count_total    count has descendants    count no descendants
    --  -----------  ---------  -----------------  -------------  -----------------------  ----------------------
     0  church        -80.6269          0.365796            4238                      242                    3996
     1  college       -57.5761          0.296741            1485                       63                    1422
     2  university    -56.1824          0.320923            1798                       85                    1713
     3  student       -46.2857          0.327337            1584                       77                    1507
     4  air           -45.095           0.132103             445                        7                     438
     5  educational   -44.5324          0.333462            1623                       81                    1542
     6  education     -38.4829          0.407449            3980                      268                    3712
     7  israel        -30.8184          0.0963412            271                        3                     268
     8  army          -30.4323          0.395735            2549                      164                    2385
     9  career        -30.3197          0.371286            1747                      102                    1645
    10  school        -27.8159          0.424283            4177                      300                    3877
    11  james         -25.804           0.270275             561                       21                     540

