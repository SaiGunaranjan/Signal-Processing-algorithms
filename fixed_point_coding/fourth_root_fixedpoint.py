# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:13:48 2023

@author: Sai Gunaranjan
"""

""" In this script, I compute fourth root of an integer in fixed point i.e X^0.25 = X^(1/4). Typically, the fourth root of a number has an integer
and a fractional part. But in fixed point processors, we cannot directly represent a fractional number. We have to use
'fractional bits' to represent a fractional number. So the implementation of fourth root in fixed point is as follows:
    1. Any number can be represented as a product of a power of 2 just smaller than X and a number which varies from 1 to 2.
    Mathematically, this can be represented as:
        X = 2^i * (1+f) ; 0 <= f < 1
        where 2^i is the power of 2 just smaller than X. When f is 0, then X = 2^i and when f reaches 1, X = 2^(i+1)
        This is also the floating point representation of a number. i is called the exponent, f is the mantissa.
        In a double precision representation of floating point numbers (64 bits),
        11 bits are reserved for the exponent(which also includes a sign bit), 52 bits for the mantissa and
        1 sign bit for the entire number. Thus making it a 64 bit floating point representation.

    2. Now, X^(1/4) = (2^i)^(1/4) * (1+f)^(1/4) implies X^(0.25) = (2^i)^0.25 * (1+f)^0.25 = 2^(i*0.25) * (1+f)^0.25
    So, to find X^(0.25) i.e. fourth root of the fixed point number, we need to find 2^(i*0.25),
    (1+f)^0.25 and then multiply these two quantities. (2^i)^0.25 varies from 1 to 2^8 and hence has 8 integer bits implicitly
    and we need to allocate some fractional bits as well. (1+f)^0.25 varies from 1 to 2^0.25 and hence has 1 integer bit implicitly and
    we need to allocate some fractional bits as well.
    Hence, to obtain the fixed point X^(0.25), we need to keep some fractional bits for both the first term as well as the second term.
    For now, lets assume we keep 10 fractional bits to represent (2^i)^0.25 as well as (1+f)^0.25.
    So, with 10 fractional bits representation, (2^i)^0.25 becomes a fixed point number with Q format 8Q10. Similarly,
    (1+f)^0.25 becomes a fixed point number with Q format as 1Q10.
    Now, we will create a look up table (LUT) for both the first term and second term with 10 fractional bits.

    LUT for (2^i)^0.25:
    For a 32 bit input number, i varies from 0 to 31. So we can create an LUT with 32 entries( of i) and 10 fractional bits
    So now, we have a framework for obtaining (2^i)^0.25.
    The LUT for (2^i)^0.25 can further be reduced from 32 entries! This is how.

    LUT for (1+f)^0.25:
    To create and store an LUT, we need some finite entries and so we will divide the f scale from 0 to 1 in steps of
    some delta (we will revisit what should be this delta a little later). And for these stepsizes, we will store the values
    of (1+f)^0.25 as 0Q10 fixed point numbers. So now, we have a framework for obtaining (1+f)^0.25. But we still need to find
    i and f for the given number X. Once we know i and f, we can plug them into their respective LUTs and get (2^i)^0.25, (1+f)^0.25
    and then we get the fourth root of the input fixed point number. Next, we will address how to obtain i, f.

    3. Find i:
        To find i, is to find the power of 2 closest to (and smaller than) X. Now, X is a fixed point number say with 32 bits.
        To find i is to find the first bit from the MSB which is 1. Thsi can be done in different ways but I follow the below approach.
        Keep right shifting X in an iterative fashion (dropping 1 LSB at a time) and keep monitoring the value of the resultant
        number. When the resultant number after dropping 1 bit at a time becomes equal to 1, then we know we have reached
        the first 1 from the MSB. We keep a counter of number of times we are dropping 1 LSB and keep checking
        for the condition that the resultant number = 1. When it converges, the value of the counter variable gives us
        the required i. This i can be used to sample the LUT of first term.


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

    5. Evaluate (1+f)^0.25:
        Once f is known, (1+f)^0.25 has to be evaluated. Note that we had stored (1+f)^0.25 as an LUT with a finite step size
        for f so that f varies from 0 to 1 is steps of delta. The f obtained from step 4, can be any value from 0 to 1 or in fixed point case,
        it can be any integer from 0 to 2^10 (since f is 0Q10). We follow the below steps to obtain the value of (1+f)^0.25:
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
               So coming back, the interval to which f belongs is computed as m = f/delta. Now delta = 2^5. So thsi essentially
               means we left shift f by 5 to get the interval m. So we now know the interval to which f belongs.
            b) Next is to evaluate (1+f)^0.25 for the actual obtained f. LUT has only the values of (1+f)^0.25 for the discrete points
               But f need not always fall on the discrete points as set in the LUT. To get the exact value of (1+f)^0.25,
               we use linear interpolation. We fit a straight line to the function (1+f)^0.25 in the interval m, m+1 and
               then evaluate the line at the f obtained in step 4. The equation of the line fitting the points x1,y1 and
               x2,y2 is given by y-y1 = slope * (x-x1), where slope = y2-y1 / x2-x1.
               x1 is the lower limit of the interval i.e. m * delta = m*2^5, y1 is the fixed point LUT value at m in 0Q10 format.
               x2 is the upper limit of the interval i.e. (m+1) * delta = (m+1)*2^5, y2 is the fixed point LUT value at m+1.
               Thus we find the true value of the (1+f)^0.25 at the obtained f.

    6. Finally find X^(1/4):
        X^(0.25) = (2^i)^0.25 * (1+f)^0.25. From step 3 we know i and from step 5 we know (1+f)^0.25. So in order to find X^(0.25),
        we just need to take product of (2^i)^0.25 and (1+f)^0.25. The first term has a format
        of 8Q10 (max power of 2 for say a 32 bit input number X) while the second term has a format 1Q10. These are two different
        Q format numbers but for multiplication, there is not need for aligning decimal/binary point and hence the 2 numbers can be multiplied as is.
        (2^i)^0.25 (8Q10) * (1+f)^0.25 (1Q10) = 9Q20. The resultant product has 20 fractioanl bits but we can drop 10
        fractional bits to bring back the resultant to 10 fractional bits. Hence the resultant X^0.25 will be of Q format 9Q10
        This is the fixed point fourth root evaluation of fixed point number X. To get the actual floating point value of X^(0.25), we have to
        divide the fixed point output by the scaling factor 2^10 (since we added 10 fraction bits to make it mQ10).

Note: Even though the input number never reaches 2**32(since uint has max value = 2**32 -1), we create an LUT for f going
from 0 to 1 including 1 (or 0 to 1024 including 1024). This is so that during interpolation, we can utilize the last interval as well.
"""
""" This is just the first implementation. I need to check if the accuracy can be improved further!"""

""" Method 2: Using smaller LUT"""
""" Lets say we wish to find the fourth root of 2^26 + 100. This can be written as 2^26 * (1+f). SO, to find the fourth root
of the number, we nned to find the fourth root of 2^26 and then the fourth root of (1+f) and then multiply these 2 terms.
Now, 2^((26)*0.25) i.e foruth root of 2^26 can be written as 2^(26/4) = 2^((24+2)/4) = 2^(24/4 + 2/4)  = 2^(6 + 2/4)
= 2^6 * 2^(2/4). We call the second term as the residual. The residual varies as 2^(0/4), 2^(1/4), 2^(2/4), 2^(3/4). We can store the LUT of these 4 numbers
instead of storing the LUT of 32 numbers
"""

import numpy as np
import sys
# import matplotlib.pyplot as plt

integerBitWidth = 32 # bits
inputFixedPointNum = np.random.randint(1,2**(integerBitWidth-1) - 1) # randint generates a signed 32 bit number in range (-2**31, 2**31 - 1)

if (inputFixedPointNum == 0):
    print('\nEstimated Fixed point X^0.25 = {}'.format(inputFixedPointNum))
    sys.exit(0)

fourthroot_inputFixedPointNum = (inputFixedPointNum)**0.25

""" Method 1"""

""" Step 2: LUT for (1+f)^0.25 """
NUM_FRAC_BITS = 10
stepSize = 1/(2**5)
f = np.arange(0,1+stepSize,stepSize) # Reason mentioned in Note
fourthroot_1pf_lut = (1+f)**0.25
fourthroot_1pf_lut_fp = np.floor((fourthroot_1pf_lut * (2**NUM_FRAC_BITS)) + 0.5).astype(np.uint32) # 1Q10

""" Step 2: LUT for 2^(i*0.25) where i varies from 0 to 31. So 32 entries for LUT"""
NUM_FRAC_BITS_1 = 10
fourthroot_2powi = 2**(np.arange(integerBitWidth)*0.25)
fourthroot_2powi_fp = np.floor((fourthroot_2powi * (2**NUM_FRAC_BITS_1)) + 0.5).astype(np.uint32) # 8Q10



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
slope = (fourthroot_1pf_lut_fp[interval+1] - fourthroot_1pf_lut_fp[interval])
mTimesXMinusX1 = slope * (resFixedPointNumNormalized - (interval<<5)) # 0Q5 * 0Q10 = 0Q15
mTimesXMinusX1 = mTimesXMinusX1 >> 5 # Drop 5 fractional bits to get 0Q10.
fourthroot_1pf_fp_eval = (fourthroot_1pf_lut_fp[interval] + mTimesXMinusX1).astype('int32') #1Q10 + 0Q10
""" Step 5"""
fourthroot_2powi_fp_eval = fourthroot_2powi_fp[nearestPowof2] # 2^(i*0.25) = 8Q10
""" Step 6"""
fourthroot_eval_fp_num = fourthroot_2powi_fp_eval * fourthroot_1pf_fp_eval # 8Q10 * 1Q10 = 9Q20
fourthroot_eval_fp_num = fourthroot_eval_fp_num >> NUM_FRAC_BITS # 9Q10
""" To obtain the equivalent floating point value of X^(0.25)"""
fourthroot_eval_float = fourthroot_eval_fp_num/(2**NUM_FRAC_BITS)


""" Method 2: Using smaller LUT"""
""" Lets say we wish to find the fourth root of 2^26 + 100. This can be written as 2^26 * (1+f). SO, to find the fourth root
of the number, we nned to find the fourth root of 2^26 and then the fourth root of (1+f) and then multiply these 2 terms.
Now, 2^((26)*0.25) i.e foruth root of 2^26 can be written as 2^(26/4) = 2^((24+2)/4) = 2^(24/4 + 2/4)  = 2^(6 + 2/4)
= 2^6 * 2^(2/4). We call the second term as the residual. The residual varies as 2^(0/4), 2^(1/4), 2^(2/4), 2^(3/4).
"""

""" Step 2: LUT for 2^(i*0.25) where i varies from 0 to 3. Only 4 entries for LUT"""
NUM_FRAC_BITS_MET2 = 10
fourthroot_2powi_smallLUT = 2**(np.arange(4)*0.25) # 4 because of 4th root
fourthroot_2powi_smallLUT_fp = np.floor((fourthroot_2powi_smallLUT * (2**NUM_FRAC_BITS_MET2)) + 0.5).astype(np.uint32) # 2Q10

""" Step 3, step 4 are same as for above method"""

integerQuotientWhenDividedby4 = nearestPowof2 >> 2 # Divide by 4 to get the nearest integer quotient (4 because we need fourth root)
""" Step 5"""
twoPowInt = 2**integerQuotientWhenDividedby4 # 7Q0 # 2**(l/4) where l is an integer multiple of 4. This can be achived through successive left shifting sequentially
residual = nearestPowof2 - (integerQuotientWhenDividedby4 << 2) # Index of Residual after factoring out the fourth root of nearest power of 2 which is a multiple of 4
fourthroot_2powi_met2_fp = twoPowInt * fourthroot_2powi_smallLUT_fp[residual] # 7Q0 * 2Q10 = 9Q10 # met2 satnds for method 2
""" Step 6"""
fourthroot_eval_met2_fp_num = fourthroot_2powi_met2_fp * fourthroot_1pf_fp_eval # 9Q10 * 1Q10 = 10Q20
fourthroot_eval_met2_fp_num = fourthroot_eval_met2_fp_num >> NUM_FRAC_BITS # 10Q10
""" To obtain the equivalent floating point value of X^(0.25)"""
fourthroot_eval_met2_float = fourthroot_eval_met2_fp_num/(2**NUM_FRAC_BITS)

print('Actual Fixed point X^0.25 = {}'.format(fourthroot_inputFixedPointNum))
print('Estimated Fixed point X^0.25 = {}'.format(fourthroot_eval_float))
print('Estimated Fixed point X^0.25 with smaller LUT = {}'.format(fourthroot_eval_met2_float))