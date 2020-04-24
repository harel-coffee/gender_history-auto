# Descendants

## Are men or women more likely to have descendants?

Women who graduated during the 1990s are more likely to have descendants and that
difference is statistically significant.
```
1980s
Men:    3755 theses,  453 with descendants. 12.06%
Women:  1869 theses,  225 with descendants. 12.04%
Fisher's Exact Test p-value: 1.000

1990s
Men:    4187 theses,  332 with descendants. 7.93%
Women:  2814 theses,  285 with descendants. 10.13%
Fisher's Exact Test p-value: 0.002

2000s
Men:    4237 theses,  164 with descendants. 3.87%
Women:  3140 theses,   96 with descendants. 3.06%
Fisher's Exact Test p-value: 0.064

```

## Are people who work on certain topics more likely to have descendants?

Yes. If we limit our dataset to dissertations finished in the 1980s and 90s, 
we find that the following topics are overrepresented among people with descendants:

``` 
topics distinctive for Corpus 1: has descendants. 1306 Documents

    topics                                        dunning    frequency_score    freq both    f has descendants    f no descendants
--  ------------------------------------------  ---------  -----------------  -----------  -------------------  ------------------
83  (45) Cultural Turn                           1808.73            0.607312   0.0208227            0.0304296           0.0196758
82  (50) 20th Century Labor History              1617.72            0.606021   0.0191446            0.0278497           0.0181054
81  (68) 19th Century African American History    780.813           0.589715   0.0134567            0.0184799           0.0128571
80  (15) Rural Social History                     764.572           0.562799   0.0287455            0.0359034           0.0278909
79  (61) Gender and Feminism                      747.34            0.56935    0.0226773            0.0289854           0.0219243
```

(note: "freq" or "f" here means topic weight. So, among people with descendants, "Cultural Turn" 
has an average topic weight of 3% whereas among those without, it has an average weight of 2%)

The following topics are overrepresented among dissertation writers without descendants.
``` 
topics distinctive for Corpus 2: no descendants. 11454 Documents

    topics                                         dunning    frequency_score    freq both    f has descendants    f no descendants
--  -------------------------------------------  ---------  -----------------  -----------  -------------------  ------------------
 0  (35) Organizations                           -1794.54            0.380896   0.0269719            0.0173042           0.028126
 1  (27) Education                               -1428.05            0.390017   0.0247961            0.0164885           0.0257878
 2  (23) Christianity                            -1263.56            0.416454   0.0363314            0.0267451           0.0374758
 3  (64) Political History of the Cold War       -1249.81            0.34846    0.0122199            0.00687666          0.0128578
 4  (31) Military History                        -1238.32            0.399632   0.0254035            0.017535            0.0263429
```

My interpretation of this is that topics that deal with race, class or gender (gender and feminism, 
labor history, African American history) lead to more descendants. And those are also areas
where women were overrepresented. Also, keep in mind that "Cultural history" is something of a
female theory topic compared to "Narratives and Discourses", its male version.

By contrast, older social history (organizations, education) and military history make it less
likely that someone would have descendants.

## Are the same topics overrepresented for men and women?
The topics that lead to more descendants look similar for men and women. The most notable change is
probably that men simply don't score highly for "Gender and Feminism" and as a result, it is not 
associated with a higher chance of having descendants.

Men: 
``` 
    topics                                    dunning    frequency_score    freq both    f has descendants    f no descendants
--  --------------------------------------  ---------  -----------------  -----------  -------------------  ------------------
83  (45) Cultural Turn                       1115.68            0.618166   0.0169842            0.0258501           0.0159673
82  (15) Rural Social History                 713.048           0.578367   0.0273911            0.036189            0.0263821
81  (50) 20th Century Labor History           609.123           0.586871   0.0186333            0.0253716           0.0178604
80  (87) Sociology and History                444.592           0.539878   0.0723526            0.0834064           0.0710848
79  (14) Colonies and Empires                 321.383           0.567565   0.0170482            0.0216785           0.0165171
```

Women: 
``` 
83  (50) 20th Century Labor History               905.621           0.6225     0.0199684            0.0306725           0.0186006
82  (68) 19th Century African American History    754.757           0.625904   0.0156076            0.0242627           0.0145016
81  (45) Cultural Turn                            676.014           0.593748   0.0274251            0.0380907           0.0260623
80  (2) Slavery in the Americas                   584.683           0.663271   0.00646452           0.0114729           0.00582455
79  (61) Gender and Feminism                      566.884           0.564421   0.0523197            0.0655972           0.0506231
```








