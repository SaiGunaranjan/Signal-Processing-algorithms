# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:08:26 2023

@author: Sai Gunaranjan Pelluri
"""

"""
In this script, I have implemented the fixed point algorithm to compute the reciprocal of a number d (1/d) using
the Newton Raphson algorithm. The Newton Raphson algorithm is an iterative algorithm which solves for the roots of a function.
So, if we want to compute a value and if we can pose the problem as a functions whose root is the value to be computed, then
we can use the Newton Raphson iterative algorithm to compute the value. The iteration step of the Newton Raphson algorithm
is derived as follows:
    1. Lets say there is a function f whose root(x) we need to find.
    2. We first start with some guess for x, say x_1. Note that the initialization value for x is critical for fast convergence.
    3. The way Newton Raphson works is as follows. We fit a line to the point x_1, f(x_1) with slope f'(x_1)and see where this line
    intersects the x axis. The point where this line intersects the x axis becomes the new x(x_2) for the next iteration.
    Now again, we fit a line through x_2, f(x_2) with slope f'(x_2) and see where this line meets the x axis. That new intersection
    point on the x axis becomes x_3 and so on. With proper initialization/seed the algorithm can converge within 3 to4 iterations
    to give the true solution x.
    4. The equation of line passing through points x1, y1 with slope m is given by:
        y-y1 = m * (x-x1)
    5. The equation of line passing through x_i, f(x_i) with slope f'(x_i) is given by:
        y-f(x_i) = f'(x_i) * (x-x_i)
        Now, the new x i.e. x_i+1 is the location where the line intersects x axis => y = 0.
        Plugging this information in the above equation we get:
            0-f(x_i) = f'(x_i) * (x_i+1 - x_i)
    6. Rearranging the above equation, we get the iterative step for the Newton Raphson algorithm as follows:
        x_i+1 = x_i - f(x_i)/f'(x_1)

Note: The initialization value for x is critical for fast convergence. If not initialized corrctly/close to the true solution, it might take longer convergence
or might not converge easily!

In this script, I compute 1/d using Newton Rapshon algorithm.
The problem of finding 1/d can be posed as follows:
    1. Let us say x = 1/d is the root of some function f.
    2. So, if we solve for the root of the function f using Newton Raphson, then we can obtain x = 1/d.
    3. How do we construct this function f?
    4. One way to construct f is f(x) = x - 1/d. But with this construction, the derivative of f(x)  i.e slope is always a constant
    and does not change with each iteration. Hence this is not a right construction of the function f.
    5. Another way of constrcution the function f is f(x) = 1/x - d. The root of this function too solves for 1/d. Also, the slope of the function
    is given by -1/x^2 and this changes with each iteration. So, we can use this function for our Newton Raphson algorithm to find the reciprocal of d
    6. With this choice of function, the update step becomes:
        x_i+1 = x_i - (1/x_i - d) / (-1/x_i^2). On rearranging this equation we get:
            x_i+1 = x_i * (2 - d*x_i)
    7. This is the update step for finding the inverse of the number d.

More material for the Newton Raphson algorithm is available here:
    https://en.wikipedia.org/wiki/Division_algorithm#:~:text=%5Bedit%5D-,Newton%E2%80%93Raphson%20division,-%5Bedit%5D

The update equation as such is straight forward and easy to implement. The challenge lies is appropriate initialization.
The range of value of d can be anything and so how do we seed the algorithm with a value(x_0) which is close to the true
1/d value. The way, we do it is we first bring d to the range of [0,1) with appropriate scaling (typically by a power of 2!)
d' = d/scale ; d > 1
   = d*scale ; d < 1

We then solve for 1/d' and then to finally get 1/d, we scale the final output with the same scaling factor used.
The reason we do this is so that the seed can now work with a finite range of value i.e from [1/0.5, 1/1).
Remember, seed is for initial guess of 1/d (1/d' in this case). Now, we can quantize the value from 0.5 to 1 in some finite steps
and can store the inverse of these values as an LUT. For fixed point implementation, we choose the number of entries of the LUT
as a power of 2.

The update equation for the scaled Newton Raphson divider becomes:
    x_i+1 = x_i * (2 - d/scale * x_i) => x_i+1 = x_i/scale * ((2*scale) - (d*x_i))

The fixed point implementation is as follows:
    1. LUT generation:
        a. We first fix the size of the LUT (defined by LUT_LENGTH) which maps [0.5,1) to [1/0.5,1/1) = [2,1), say 32 (typically this is chosen as a power of 2)

        b. log2(LUT_LENGTH) gives the bitwidth of each indexing entry of the LUT. So for a 32 length LUT, each entry will be 5 bits from 00000 to 11111

        c. Since we will be operating on the scaled number(d' = d/scale) for computing the inverse, d' will lie in [0.5,1).
        Hence, post scaling, d' will be greater than 0.5 but less than 1.
        This means that the binary representation of d' will not have any integer bits and will only have fractional bits and will
        be of the form 0.1xxxxxx. This is because post scaling, d' will be greater than 0.5 but less than 1.
        The number of x's + 1(1 additional bit for 0.5) is the total fractional bits of d' post scaling.

        d. Since 0.5 is common for numbers from [0.5,1), we can subtract 0.5 from all of them and make them from 0 to 0.5

        e. Hence, for generating the LUT, we take log2(LUT_LENGTH) number of bits i.e. 5 bits(in this case) from the first occurence of 1
        from the MSB side. So 5 bits to the right of the first occurence of 1 from the MSB side.

        f. Note that all these bits are post the decimal point and post a 1.

        g. So, the weight of the MSB bit will be 2^-2 and the weight of the LSB will be 2^-6. This is because, there is already a 1 to
        the immediate right of the decimal post whose weight is 2^-1. From this, we are taking 5 bits and
        hence the weight of all the remaining 5 bits will start from 2^-2 to 2^-6.

        h. These are the values corresponding to the 32, 5 bit entries going from 00000 to 11111.

        i. So, 00000 maps to 0.5
               00001 maps to 0.5 + 2^-6
               00010 maps to 0.5 + 2^-5
               00011 maps to 0.5 + 2^-5 + 2^-6
               .
               .
               .
               11111 maps to 0.5 + 2^-2 + 2^-3 + 2^-4 + 2^-5 + 2^-6

        Now, we need to store the inverse of these values as a seed.
        These inverse values are stored with fractional bitwidth = ONE_BY_DEN_FRACBITWIDTH (16 or 10 in our case).
        So, the mapping  now becomes
               00000 which is 0.5 is mapped to 1/0.5
               00001 which is 0.5 + 2^-6 is mapped to 1/(0.5 + 2^-6)
               00010 which is 0.5 + 2^-5 is mapped to 1/(0.5 + 2^-5)
               00011 which is 0.5 + 2^-5 + 2^-6 is mapped to 1/(0.5 + 2^-5 + 2^-6)
               .
               .
               .
               11111 which is 0.5 + 2^-2 + 2^-3 + 2^-4 + 2^-5 + 2^-6 is mapped to 1/(0.5 + 2^-2 + 2^-3 + 2^-4 + 2^-5 + 2^-6)
        This is how the LUT is generated and stored.

        j. Any fixed point number we have to invert will have some integer part and a fractional part.
        So, we first scale the number to bring it to the range [0.5,1). This is done easily in fixed point by moving the
        decimal point to the left of the 1st occurence of a 1 from the MSB. But, we don't have to exlicitly do this.
        This will be taken care of by computing the scaling factor/shift. We find the position of the leading 1 from the
        LSB side and subtract it with the number of fractional bits. This gives the scaling factor/shift.
        Ex: Consider a binary fixed point number 000100101.1011010100110110,. To bring it the range [0.5,1),
        we need to shift the decimal point to the left of the leading 1 i.e 000.1001011011010100110110. Now the number lies between [0.5,1)
        and the scaling shift is position of leading 1 from LSB i.e 22 - number of fractional bits = 16. So, the scaling shift
        is 22 - 16 = 6 or the scaling factor is 2^6. If we look at it from decimal point of view, the number 000100101.1011010100110110
        is 37.decimalpart. Now, to bring 37.something to [0.5,1), we need to divide by a number which is a nearest power of 2 i.e
        64 which is 2^6. Hence it makes sense. The same holds true even if the leading one occurs in the fractional part i.e after the decimal point
        This way, we get the scaling shift/factor.

        k. Now, once we have found the scaling shift (and hence the position of the leading 1 from LSB), we know that we are implicitly working with
        a number of the form 0.1xxxx, where x's can be any of 0/1. So, the number is brought to our required range of [0.5,1).
        Now, we extract the LUT_INDEXING_BITWIDTH number of bits from the 1st occurence of leading 1 (from LSB) excluding that 1.
        Ex: For the number we had taken, xxx100101.1011010100110110, post scaling, it becomes xxx.1001011011010100110110. So we now have to extract
        LUT_INDEXING_BITWIDTH (=5 in our example) bits from 1 onwards i.e 00101. Now, we index this entry into our LUT and get the
        corresponding seed value for 1/d' (Note that d' is the scaled verion of d). This becomes our x_0.

        l. We apply the update equation:
            x_i+1 = x_i * (2 - d/scale * x_i) => x_i+1 = x_i/scale * ((2*scale) - (d*x_i))
        So, we first multiply the denominator d with x_0. d has  DEN_FRAC_BITS fractional bits, x_0 has  ONE_BY_DEN_FRACBITWIDTH fractional bits.
        The resultant multiplication yields DEN_FRAC_BITS +  ONE_BY_DEN_FRACBITWIDTH fractional bits.

        m. This product has to subtracted from 2. But before this, we need to align the decimal point and also scale 2 with the scaling factor/shift
        So 2 is left shifted by DEN_FRAC_BITS +  ONE_BY_DEN_FRACBITWIDTH bits and then also left/right shifted by scale depending on whether scale shift is +/-.
        Scale shift will be +ve if the original number was >1 and it will be -ve if the original number was itself <1

        n. Subtract the scaled 2 and d*x_0. The result still has DEN_FRAC_BITS +  ONE_BY_DEN_FRACBITWIDTH fractional bits.

        o. Apply the scaling shift to the above result.

        p. Multiply the result from above step with x_0. This will result in  DEN_FRAC_BITS +  ONE_BY_DEN_FRACBITWIDTH +  ONE_BY_DEN_FRACBITWIDTH
        fractional number of bits.

        q. Drop DEN_FRAC_BITS +  ONE_BY_DEN_FRACBITWIDTH bits from the above result else it will overflow.

        r. Repeat the above steps from l through p for NR_ITER iterations. Then we get the reciprocal of the scaled denominator i.e 1/d'

        s. But we need 1/d, so scale the result from above step with the scale factor. Now we obtain 1/d in fixed point with
        DEN_FRAC_BITS fractional bits.

        t. To obtain the equivalent floating point number, divide by 2**DEN_FRAC_BITS

More material for the fixed point divider using Newton Raphson algorithm is available here:
    https://docs.google.com/document/d/16j8StyirLyQ6gH1IHuCZEE-NPAUjVRiK/edit

The script has been modified to cater to the smart RADAR divider where in the numerator is 8Q8 and denominator is 8Q8.
Hence now, I can afford to accomodate bit growth upto the final stage and then drop bits only at the end.

"""

import numpy as np
import sys

# np.random.seed(10)

LUT_LENGTH = 32 # Ideal to have it as a power of 2
LUT_INDEXING_BITWIDTH = int(np.log2(LUT_LENGTH))
ONE_BY_DEN_FRACBITWIDTH = 16#10

ONE_BY_DEN_LUT_FLOAT = 1/(np.arange(0.5,1,2**-(LUT_INDEXING_BITWIDTH+1)))
ONE_BY_DEN_LUT = (np.floor((ONE_BY_DEN_LUT_FLOAT * (2**ONE_BY_DEN_FRACBITWIDTH)) + 0.5)).astype(np.int32)

OUTPUT_FRAC_BITS = 16#16

NUM_TOT_BITS = 16
NUM_FRAC_BITS = 8
NUM_SIGN_BIT = 0
NUM_INT_BITS = NUM_TOT_BITS - NUM_FRAC_BITS + NUM_SIGN_BIT

numFloat = np.random.randint(0,2**(NUM_INT_BITS)-1) + np.random.random() # Failure case for ONE_BY_DEN_FRACBITWIDTH = 16 : 27881.24593943644
numFixed = np.floor((numFloat * 2**NUM_FRAC_BITS) + 0.5).astype(np.int64)

DEN_TOT_BITS = 16
DEN_FRAC_BITS = 8#20#16
DEN_SIGN_BIT = 0
DEN_INT_BITS = DEN_TOT_BITS - DEN_FRAC_BITS + DEN_SIGN_BIT



# denFloat = 1.965#100.997652#1.965#15.375
denFloat = np.random.randint(0,2**(DEN_INT_BITS)-1) + np.random.random() # Failure case for ONE_BY_DEN_FRACBITWIDTH = 16 : 27881.24593943644
denFixed = np.floor((denFloat * 2**DEN_FRAC_BITS) + 0.5).astype(np.int64)

NR_ITER = 3 # Newton-Raphson Iteration

if (denFixed == 0):
    print('Cannot compute 1/0 !')
    sys.exit()

num = denFixed
count = 0
while (num!=0):
    num = (num >> 1)
    count += 1

positionOfLeading1fromLSB = count

# numLeadingZeros = DEN_TOT_BITS - positionOfLeading1fromLSB
# scalingFactor = DEN_INT_BITS - numLeadingZeros # If positive, right shift else left shift

scalingFactor = positionOfLeading1fromLSB - DEN_FRAC_BITS # Scaling factor/shifts can be computed this way. If positive, right shift else left shift

""" Step k. Extract the index to sample into the LUT"""
if (positionOfLeading1fromLSB <  (LUT_INDEXING_BITWIDTH+1)):
    indexToSampleLUT = (denFixed << ((LUT_INDEXING_BITWIDTH+1) - positionOfLeading1fromLSB)) & (2**LUT_INDEXING_BITWIDTH - 1)
else:
    indexToSampleLUT = (denFixed >> (positionOfLeading1fromLSB - (LUT_INDEXING_BITWIDTH+1))) & (2**LUT_INDEXING_BITWIDTH - 1)

denDownScaled = np.int64(denFixed) # Denominator is actually of format DEN_INT_BITS Q DEN_FRAC_BITS but treated as 0 Q scalingFactor+DEN_FRAC_BITS
x = ONE_BY_DEN_LUT[indexToSampleLUT] # This is the initial 1/d where d is downscaled to lie in [0.5,1)


for ele in range(NR_ITER):
    """
    x = x * (2-(den/scaling)*x)
    x = x/scaling * ((2*scaling) - (den*x))
    """
    a1 = (denDownScaled * x) #  DEN_FRAC_BITS * 2QONE_BY_DEN_FRACBITWIDTH # Not dropping any fractional bits post d*x

    """Bring 2 to the same decimal scale as a1"""
    a2 = 2 << (DEN_FRAC_BITS + ONE_BY_DEN_FRACBITWIDTH)

    if (scalingFactor>=0):
        """ Multiply 2 (a2) with the scaling factor if scaling factor is +ve (>1)"""
        a2 = a2 << scalingFactor #
    else:
        """ Divide 2 (a2) with the scaling factor if scaling factor is -ve (<1)"""
        a2 = a2 >> np.abs(scalingFactor)

    """ There will be a bit growth post multiplication of den and x and also for the scaling of 2.
    If it is contained to only 32 bit, it will overflow, hence containing the below result in a 64 bit datatype"""
    a3 = (a2 - a1).astype(np.int64)

    a4 = (x * a3) >> (DEN_FRAC_BITS+ONE_BY_DEN_FRACBITWIDTH) # Bring back the result to ONE_BY_DEN_FRACBITWIDTH

    if  (scalingFactor >=0):
        """Divide x by the scaling factor if scaling factor is +ve (>1) """
        a4 = a4 >> scalingFactor
    else:
        """ Multiply x by the scaling factor if scaling factor is -ve (<1)"""
        a4 = a4 << np.abs(scalingFactor)

    x = a4 # fractional bitwidth of x = ONE_BY_DEN_FRACBITWIDTH
    # print('Iter # = {}'.format(ele))

"""Bring back the scaling factor to the final output to get the true value of 1/d """
if (scalingFactor >=0):
    xRescaled = x >> scalingFactor
else:
    xRescaled = x << np.abs(scalingFactor)

newtonRaphsonBased1byDenFloat = xRescaled/(2**(ONE_BY_DEN_FRACBITWIDTH))
true1byDen = 1/denFloat

print('Float 1/d = {}'.format(true1byDen))
print('Fixed 1/d = {}'.format(newtonRaphsonBased1byDenFloat))

numByDenFloat = numFloat * true1byDen

# NUM_INT_BITSQ(NUM_FRAC_BITS) * zQ(ONE_BY_DEN_FRACBITWIDTH) --> (NUM_INT_BITS+z)Q(OUTPUT_FRAC_BITS)
numByDenFixed = (numFixed * xRescaled) >> (NUM_FRAC_BITS + ONE_BY_DEN_FRACBITWIDTH - OUTPUT_FRAC_BITS) #  NUM_FRAC_BITS + ONE_BY_DEN_FRACBITWIDTH - OUTPUT_FRAC_BITS
numByDenFixedConvertFloat = numByDenFixed/(2**OUTPUT_FRAC_BITS)
print('Float n/d = {}'.format(numByDenFloat))
print('Fixed n/d = {}'.format(numByDenFixedConvertFloat))
