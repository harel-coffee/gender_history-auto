

## What percentage of men and women are advised by men and women?

Important to note here as well is the huge number of unknown advisors in the 1980s, i.e.
advisor data is simply not available for those. Given that the number of female-advised theses is 
tiny for the 1980s, I think it's fine to ignore the 1980s.

``` 
1980s
Men:    416 male advisors,   35 female advisors. 3304 unknown advisors. 7.76% female advisors
Women:  194 male advisors,   55 female advisors. 1620 unknown advisors. 22.09% female advisors

1990s
Men:   2759 male advisors,  375 female advisors. 1053 unknown advisors. 11.97% female advisors
Women: 1499 male advisors,  627 female advisors.  688 unknown advisors. 29.49% female advisors

2000s
Men:   2908 male advisors,  701 female advisors.  628 unknown advisors. 19.42% female advisors
Women: 1639 male advisors, 1055 female advisors.  446 unknown advisors. 39.16% female advisors

2010s
Men:    445 male advisors,  148 female advisors.  108 unknown advisors. 24.96% female advisors
Women:  295 male advisors,  238 female advisors.   84 unknown advisors. 44.65% female advisors
```

## Advisors have a huge impact on topic selection

Here are the top 5 overrepresented topics for students with a female advisor for
dissertations submitted from 1990 to 2015:
``` 
topics distinctive for Corpus 1: female advisor. 3180 Documents

    topics                                        dunning    frequency_score    freq both    f female advisor    f male advisor
--  ------------------------------------------  ---------  -----------------  -----------  ------------------  ----------------
83  (61) Gender and Feminism                    27239.2             0.74463    0.0265462            0.0525673        0.0180279
82  (76) Consumption and consumerism             5388.89            0.686677   0.0101602            0.0172094        0.00785249
81  (45) Cultural Turn                           4063.38            0.588702   0.040316             0.0521571        0.0364396
80  (32) Doctors & Patients                      3279.8             0.667168   0.0080016            0.0128547        0.00641287
79  (46) Family                                  2678.49            0.632007   0.0111629            0.0162894        0.00948466
78  (71) Sexuality                               2553.61            0.670565   0.00594615           0.0096412        0.00473653

```

And the ones for students with male advisors:

``` 
topics distinctive for Corpus 2: male advisor. 9633 Documents

    topics                                                 dunning    frequency_score    freq both    f female advisor    f male advisor
--  ---------------------------------------------------  ---------  -----------------  -----------  ------------------  ----------------
 0  (31) Military History                                -4926.9             0.362628   0.0273217           0.0173936         0.0305718
 1  (64) Political History of the Cold War               -3896.08            0.281979   0.0092391           0.0042675         0.0108666
 2  (33) Political History (Revolutions)                 -1738.33            0.452279   0.0722267           0.0623191         0.0754701
 3  (66) German and Austro-Hungarian Diplomatic History  -1553.39            0.269658   0.00333475          0.00145809        0.00394909
 4  (43) Islamic History                                 -1214.74            0.36823    0.00728225          0.00473125        0.00811736
```
(Political history topics like U.S. political and 19th century U.S political follow soon after.)

## Are these trends stable over time?

Largely yes. Here's just the 1990s:
``` 
    topics                                        dunning    frequency_score    freq both    f female advisor    f male advisor
--  ------------------------------------------  ---------  -----------------  -----------  ------------------  ----------------
83  (61) Gender and Feminism                    16798               0.772654   0.0279099           0.0645204         0.0189845
82  (76) Consumption and consumerism             3718.49            0.728622   0.00990729          0.0199963         0.00744768
81  (46) Family                                  2019.77            0.669114   0.0113589           0.0191359         0.00946297
80  (45) Cultural Turn                           1811.35            0.601466   0.0326593           0.0448164         0.0296956
79  (32) Doctors & Patients                      1483.69            0.674881   0.00769971          0.0131997         0.00635886
```

And here's 2005-2015:
``` 
    topics                                   dunning    frequency_score    freq both    f female advisor    f male advisor
--  -------------------------------------  ---------  -----------------  -----------  ------------------  ----------------
83  (61) Gender and Feminism                5522.21            0.713269   0.0234938            0.0400034        0.0160812
82  (76) Consumption and consumerism        1459.84            0.673363   0.00996857           0.0154639        0.00750125
81  (71) Sexuality                           950.479           0.665413   0.00720878           0.0109742        0.00551814
80  (32) Doctors & Patients                  919.858           0.648616   0.00884451           0.0129355        0.00700772
79  (45) Cultural Turn                       778.505           0.56192    0.048043             0.056661         0.0441737
```

Family and sexuality trade spots.

## Female advisors have a huge influence on their male advisees

This is probably the most interesting finding in this document: female advisors have 
an immense influence on the topics of their male advisees. 

Ok, that seems kind of obvious but one one simple counter argument to the data presented 
above would be: Sure, students of female advisors work more on gender but that's no surprise
because female advisors have more female students than male advisors. 

But that's not the case. If we limit our dataset to male students, those with female 
advisors are more likely to write about the following topics.

I think that's important to highlight because it allows us to show second-order effects.
Women becoming history professors doesn't just change the kind of work being done because women do
different work. Their influence continues on through their students, both male and female.

``` 
topics distinctive for Corpus 1: man with female advisor. 1224 Documents

    topics                                         dunning    frequency_score    freq both    f man with female advisor    f man with male advisor
--  -------------------------------------------  ---------  -----------------  -----------  ---------------------------  -------------------------
83  (61) Gender and Feminism                      2029.04            0.695186   0.00728158                   0.013713                   0.00601268
82  (45) Cultural Turn                            1464.28            0.586991   0.0342695                    0.0455441                  0.0320451
81  (39) Music                                    1163.53            0.613988   0.0149472                    0.0216663                  0.0136216
80  (71) Sexuality                                 627.969           0.643195   0.00477405                   0.00760066                 0.00421637
79  (68) 19th Century African American History     497.899           0.572548   0.0172614                    0.0218959                  0.016347
```

And those with male advisors are more likely to write about the following topics:
``` 
topics distinctive for Corpus 2: man with male advisor. 6112 Documents

    topics                                                 dunning    frequency_score    freq both    f man with female advisor    f man with male advisor
--  ---------------------------------------------------  ---------  -----------------  -----------  ---------------------------  -------------------------
 0  (31) Military History                                -2989.88            0.351664   0.035891                     0.0210553                  0.038818
 1  (64) Political History of the Cold War               -1886.06            0.292607   0.0124617                    0.00570602                 0.0137946
 2  (66) German and Austro-Hungarian Diplomatic History   -645.916           0.297475   0.00445622                   0.00208502                 0.00492405
 3  (43) Islamic History                                  -606.643           0.360825   0.00818095                   0.00497531                 0.00881341
 4  (74) U.S. Political Parties & Elections               -403.333           0.393597   0.00889138                   0.00612532                 0.00943712
```

One note on interpretation: "Gender and Feminism" has a frequency score of 0.69. This means that, 
on average, a man with a female advisor has a "Gender and Feminism" weight twice as large as a 
man with a male advisor. (actually, it's even slightly larger. a 2:1 ratio would lead to a frequency
score of 0.67)

## The same finding holds true for female students

The results for female students look very similar:

``` 
topics distinctive for Corpus 1: woman with female advisor. 1920 Documents

    topics                                        dunning    frequency_score    freq both    f woman with female advisor    f woman with male advisor
--  ------------------------------------------  ---------  -----------------  -----------  -----------------------------  ---------------------------
83  (61) Gender and Feminism                    9285.32             0.661541   0.0528513                      0.0769934                    0.0393916
82  (76) Consumption and consumerism            3583.4              0.691015   0.0141213                      0.0218921                    0.00978895
81  (71) Sexuality                              1353.11             0.66302    0.00754997                     0.0110336                    0.00560782
80  (32) Doctors & Patients                     1347.12             0.626553   0.0129503                      0.0174855                    0.0104219
79  (45) Cultural Turn                          1107.53             0.561035   0.0486447                      0.0565436                    0.0442409

topics distinctive for Corpus 2: woman with male advisor. 3433 Documents

    topics                                      dunning    frequency_score    freq both    f woman with female advisor    f woman with male advisor
--  ----------------------------------------  ---------  -----------------  -----------  -----------------------------  ---------------------------
 0  (58) Soviet Union                          -930.113           0.320516   0.00553542                    0.00322                       0.00682631
 1  (70) 20th Century British Foreign Policy   -729.455           0.236133   0.00207159                    0.000850785                   0.00275221
 2  (33) Political History (Revolutions)       -617.29            0.457571   0.0608006                     0.0543313                     0.0644073
 3  (20) Markets & Trade                       -552.985           0.429699   0.0202122                     0.0167031                     0.0221685
 4  (64) Political History of the Cold War     -552.627           0.350177   0.00465567                    0.00300482                    0.00557604
```

## Final Note

Still, though, it's worth noting that average gender and feminism weights remain skewed:

- woman, female advisor: 7.7%
- woman, male advisor: 3.9%
- man, female advisor: 1.4%
- man, male advisor: 0.6%

So, a female advisor, on average, doubles the weight for the gender and feminism topic compared
to a male advisor--it's just that the male baseline is very low.


