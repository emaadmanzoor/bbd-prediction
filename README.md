Benign Breast Disease Prediction
================================

Builds a classifier using regularized logistic regression on
matched control-case data from studies on the risk factors associated
with benign breast disease.

## Building The Model

   * *Algorithm*: Logistic regression
 
   * *Enhancements*: 
      * Quadratic features were generated from the 12 features in the data set.
      * The features were regularized, and the accuracy measure on the data set
        touched 95.5% at a factor of 1.0, increasing as the factor was decreased.
        The increasing accuracy with decreasing regularization is most probably
        an indicator of over-fitting the current data.

## About The Data

   * *Name*: Benign Breast Disease 1-3 Matched Case-Control Study (BBDM13.DAT)
   * *Source*: [University of Massachusetts - Amherst](http://www.umass.edu/statdata/statdata/stat-logistic.html)
   * *Size*: 200 observations, 14 variables
   * *Type*: Matched case-control
   * *Original Source*: These data come from Hosmer and Lemeshow (2000) Applied Logistic
 	                    Regression: Second Edition, page 245. These data are copyrighted
                        by John Wiley & Sons Inc. and must be acknowledged and used accordingly.

   Further information can be found in *bbdm13.txt*.

## References

   1.   Pastides, H., Kelsey, J.L., Holford, T.R., and LiVolsi, V.A., (1985).
     The epidemiology of fibrocystic breast disease.  American Journal of 
     Epidemiology, 121, 440-447.

   2.   Pastides, H., Kelsey, J.L., LiVolsi, V.A., Holford, T., Fischer, D., 
     and Goldberg, I.(1983).  Oral contraceptive use and fibrocystic breast
     disease with special reference to it histopathology.  Journal of the
     National Cancer Institute, 71, 5-9.

   3.   Hosmer and Lemeshow, Applied Logistic Regression, Wiley, (1989). 
