# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 00:48:10 2023

@author: Sai Gunaranjan
"""

""" In this script, I perform log2 computation in fixed point. Typically, the log2 of a number has an integer
and a fractional part. But in fixed point processors, we cannot directly represent a fractional number. We have to use
'fractional bits' to represent a fractional number. So the implementation of log2 in fixed point is as follows:
    1. Any number can be represented as a product of a power of 2 just smaller than X and a number which varies from 1 to 2.
    Mathematically, this can be represented as:
        X = 2^i * (1+f) ; 0 <= f < 1
        where 2^i is the power of 2 just smaller than X. When f is 0, then X = 2^i and when f reaches 1, X = 2^(i+1)
        This is also the floating point representation of a number. i is called the exponent, f is the mantissa.
        In a double precision representation of floating point numbers (64 bits),
        11 bits are reserved for the exponent(which also includes a sign bit), 52 bits for the mantissa and
        1 sign bit for the entire number. Thus making it a 64 bit floating point representation.

    2. Now, log2(X) = i + log2(1+f)
    So, to find log2(X) i.e. log2 of the fixed point number, we need to find i (the power of 2 closest to X),
    log2(1+f) and then sum these two quantities. i is an integer while log2(1+f) is a fractional which varies from 0 to 1
    Hence, to obtain the fixed point log2(X), we need to keep some fractional bits for log2(1+f).
    For now, lets assume we keep 10 fractional bits to represent log2(1+f). So, with 10 fractional bits representation,
    log2(1+f) essentially becomes a fixed point number with 10 fractional bits with the Q format as 0Q10.
    So, in fixed point format, log2(1+f) becomes log2(1+f)*2^10 and hence the value ranges from 0 to 2^10.
    Now, we will create a look up table (LUT) for this quantity log2(1+f) in fixed point notation.
    To create and store an LUT, we need some finite entries and so we will divide the f scale from 0 to 1 in steps of
    some delta (we will revisit what should be this delta a little later). And for these stepsizes, we will store the values
    of log2(1+f) as 0Q10 fixed point numbers. SO now, we have a framework for obtaining log2(1+f). But we still need to find
    i and f for the given number X. Once we know i and f, we can plug f into the LUT and get log2(1+f) and then we get the log2
    of the input fixed point number. Next, we will address how to obtain i, f.

    3. Find i:
        To find i, is to find the power of 2 closest to (and smaller than) X. Now, X is a fixed point number say with 32 bits.
        To find i is to find the first bit from the MSB which is 1. This can be done in different ways but I follow the below approach.
        Keep right shifting X in an iterative fashion (dropping 1 LSB at a time) and keep monitoring the value of the resultant
        number. When the resultant number after dropping 1 bit at a time becomes equal to 1, then we know we have reached
        the first 1 from the MSB. We keep a counter of number of times we are dropping 1 LSB and keep checking
        for the condition that the resultant number = 1. When it converges, the value of the counter variable gives us
        the required i.


    4. Find f:
        We have X and from above step, we have got i. Coming back to our original floating point representation of X, we have
        X = 2^i * (1+f). This implies, X/(2^i) - 1 = f. So, to obtain f, we need to divide X by 2^i and then subtract 1.
        The resultant f is a number always between 0 and 1. But remember, we are working on a fixed point machine
        and hence we cannot have any fractional numbers. So how do we circumvent this? To avoid the fractional numbers for f,
        we will have a fixed point representation for f as well. Lets say we reserve 10 fractional bits for f as well. This means,
        we will multiply with 2^10 on both sides. So the fixed point representation for f becomes:
            f * 2^10 = (X/2^i) * 2^10 - 2^10. So to find the value of f in fixed point (with 10 fracrional bits), we need to do the following
            a) The first term is X/2^i * 2^10. This operation is essentially multiplication and division with powers of 2. This can easily
               be accomplished with left shifts and right shifts (for division and multiplication respectively).
            b) The first term can be represented as:
                                                    X * 2^(10-i) if i < 10,
                                                    X / 2^(i-10) if i > 10
            So based on whether i is greater than or less than 10 (fractional bits used to represent f),
            we either do a right shift by 10-i or a left shift by i-10.
            Note that there is also a subtraction by 2^10 post the above step. So the above result is subject to a
            subtraction with 1 << 10. Thus we obtain the value of f in 10 fractional bits notation.

    5. Evaluate log2(1+f):
        Once f is known, log2(1+f) has to be evaluated. Note that we had stored log2(1+f) as an LUT with a finite step size
        for f so that f varies from 0 to 1 is steps of delta. The f obtained from step 4, can be any value from 0 to 1 or in fixed point case,
        it can be any integer from 0 to 2^10 (since f is 0Q10). We follow the below steps to obtain the value of log2(1+f):
            a) Find out the step size interval into which the f falls.
               Since we have quantized the f interval while storing the LUT, we first need to find out the interval into
               which the evaluated f falls. The interval is found out as follows. Since each interval size is delta,
               m * delta <= f < (m+1) * delta, for some m. So we find the closest integer m such that m * delta <= f.
               Therefore m = f/delta. So in order to find  m, we need to divide f by delta. Now, f is fixed point integer
               with 10 fractional bits. Fixed point division is not straight forward when the divisor (delta in this case)
               is not a power of 2. But when the divisor is a power of 2 then it is very easy. So if we can convert the delta
               into a power of 2 then the division becomes just a left shift. So our objective would now be to somehow
               make this delta a power of 2. By construction, we have divided the interval from 0 to 1 in steps of
               delta and hence delta is less than 1. So, delta is also a fractional. In order to convert it to a power of 2 we do the following.
               i) First we make delta 1/(2^p) for some positive power p. Lets say we choose p = 5, then it means the value of delta
               is 1/32. So there will be 32 intervals of size 1/32 from 0 to 1. Now, lets make f also a fixed point number with 10
               fractional bits. So f becomes f * 2^10. So f now goes from 0 to 2^10. So the step size delta also gets scaled by
               2^10. So delta becomes 1/(2^p) * 2^10. If p = 32, then delta becomes 2^5. So in our new scale, we have
               delta = 2^5 and f goes from 0 to 2^10. Note that we still have 2^5 = 32 intervals. Now we have made delta a power of 2.
               So coming back, the interval to which f belongs is computed as m = f/delta. Now delta = 2^5. So this essentially
               means we left shift f by 5 to get the interval m. So we now know the interval to which f belongs.
            b) Next is to evaluate log2(1+f) for the actual obtained f. LUT has only the values of log2(1+f) for the discrete points
               But f need not always fall on the discrete points as set in the LUT. To get the exact value of log2(1+f),
               we use linear interpolation. We fit a straight line to the function log2(1+f) in the interval m, m+1 and
               then evaluate the line at the f obtained in step 4. The equation of the line fitting the points x1,y1 and
               x2,y2 is given by y-y1 = slope * (x-x1), where slope = y2-y1 / x2-x1.
               x1 is the lower limit of the interval i.e. m * delta = m*2^5, y1 is the fixed point LUT value at m in 0Q10 format.
               x2 is the upper limit of the interval i.e. (m+1) * delta = (m+1)*2^5, y2 is the fixed point LUT value at m+1.
               Thus we find the true value of the log2(1+f) at the obtained f.

    6. Finally find log2(X):
        log2(X) = i + log2(1+f). From step 3 we know i and from step 5 we know log2(1+f). So in order to find log2(X),
        we just need to sum i and log2(1+f). But we cannot directly sum these two because i has a format
        of 5Q0 (max power of 2 for say a 32 bit input number X) while log2(1+f) has a format 0Q10. These are two different
        Q format numbers and hence cannot be added directly. We need to align the decimal point. For this,
        we need to either drop 10 fractional bits of log2(1+f) or add 10 fractional bits to i to make it 5Q10.
        Dropping 10 fractional bits for log2(1+f) defeats the entire purpose and hence the obvious way is to add 5 fractional bits to i.
        This is achieved by left shifting i by 10. So i now becomes 5Q10 and log2(1+f) is 0Q10. Now the decimal points are
        aligned and hence can be added. Therefore the final log2(X) = i << 10 + log2(1+f). This is the fixed point
        log2 evaluation of fixed point number X. To get the actual floating point value of log2(X), we have to
        divide the fixed point output by the scaling factor 2^10 (since we added 10 fraction bits to make it mQ10).

Note: Even though the input number never reaches 2**32(since uint has max value = 2**32 -1), we create an LUT for f going
from 0 10 1 including 1 (or 0 to 1024 including 1024). This is so that during interpolation, we can utilize the last interval as well.
"""


import numpy as np
# import matplotlib.pyplot as plt


inputFixedPointNum = np.random.randint(0,2**31 - 1) # randint generates a signed 32 bit number in range (-2**31, 2**31 - 1)
log2_inputFixedPointNum = np.log2(inputFixedPointNum)

""" Step 2: LUT for log2(1+f) """
NUM_FRAC_BITS = 10
stepSize = 1/(2**5)
f = np.arange(0,1+stepSize,stepSize) # Reason mentioned in Note
log2_1pf_lut = np.log2(1+f)
log2_1pf_lut_fp = np.floor((log2_1pf_lut * (2**NUM_FRAC_BITS)) + 0.5).astype(np.uint32)

num = inputFixedPointNum
""" Step 3"""
count = 0
while (num != 1):
    num = (num >> 1)
    count += 1

nearestPowof2 = count
""" Step 4"""
if (NUM_FRAC_BITS >= nearestPowof2):
    resFixedPointNum = inputFixedPointNum << (NUM_FRAC_BITS-nearestPowof2)
else:
    resFixedPointNum = inputFixedPointNum >> (nearestPowof2-NUM_FRAC_BITS)
resFixedPointNumNormalized = resFixedPointNum - (1<<NUM_FRAC_BITS) # 0Q10
interval = resFixedPointNumNormalized >> 5
slope = (log2_1pf_lut_fp[interval+1] - log2_1pf_lut_fp[interval]) #>> 5
# What is the Q format of slope. Slope is 5 bit number.
mTimesXMinusX1 = slope * (resFixedPointNumNormalized - (interval<<5)) # 0Q5 * 0Q10 = 0Q15
mTimesXMinusX1 = mTimesXMinusX1 >> 5 # Drop 5 fractional bits to get 0Q10.

""" Step 5"""
log2_1pf_fp_eval = (log2_1pf_lut_fp[interval] + mTimesXMinusX1).astype('int32')

""" Step 6"""
log2_eval_fp_num = (nearestPowof2<<NUM_FRAC_BITS) + log2_1pf_fp_eval

""" To obtain the equivalent floating point value of log2(X)"""
log2_eval_float = log2_eval_fp_num/(2**NUM_FRAC_BITS)

print('Actual Fixed point log2 = {}'.format(log2_inputFixedPointNum))
print('Estimated Fixed point log2 = {}'.format(log2_eval_float))

