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

The update equation as such is straight forward and easy to implement. The challenge lies in appropriate initialization.
The range of value of d can be anything and so how do we seed the algorithm with a value(x_0) which is close to the true
1/d value. The way, we do it is we first bring d to the range of [0.5,1) with appropriate scaling (typically by a power of 2!)
d' = d/scale ; d > 1
   = d*scale ; d < 1

We then solve for 1/d' and then to finally get 1/d, we scale the final output with the same scaling factor used.
The reason we do this is so that the seed can now work with a finite range of value i.e from [1/0.5, 1/1).
Remember, seed is for initial guess of 1/d (1/d' in this case). Now, we can quantize the value from 0.5 to 1 in some finite steps
and can store the inverse of these values as an LUT. For fixed point implementation, we choose the number of entries of the LUT
as a power of 2.

The update equation for the scaled Newton Raphson divider becomes:
    x_i+1 = x_i * (2 - d/scale * x_i) => x_i+1 = x_i/scale * ((2*scale) - (d*x_i))
If d' is the scaled denominator, then the update equation is:
    x_i+1 = x_i * (2 - d' * x_i)

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
        and the scaling shift is position of leading 1 from LSB i.e 22 - 16 (number of fractional bits). So, the scaling shift
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

        l. Scaling the denominator:
            Denominator is actually of format DEN_INT_BITS Q DEN_FRAC_BITS but brought to 0 Q NR_FRAC by discarding the
            leading zeros and taking all the bits from the 1st 1 (from the MSB) since this is where the content is.
            Also, this is how the data will be post applying the scaling factor and bringing it to range [0.5,1).
            In this implementation, we want the value to have NR_FRAC bits and hence we left shift to add some bits
            so that the final value is of format 0QNR_FRAC. Note: We are assuming it is brought to range [0.5,1) and
            hence no integer bits.

        m. Bringing x from ONE_BY_DEN_FRACBITWIDTH to NR_FRAC bits in the first iteration itself to improve final prcision


        n. We apply the update equation:
            x_i+1 = x_i * (2 - d' * x_i)
        So, we first multiply the scaled denominator d with x_0. d has DEN_FRAC_BITS fractional bits and scaled d i.e d' has NR_FRAC bits with 1 as the MSB,
        x_0 has ONE_BY_DEN_FRACBITWIDTH fractional bits and post above step, x_0 has NR_FRAC bits.
        The resultant multiplication yields NR_FRAC +  NR_FRAC fractional bits. We then drop NR_FRAC bits to bring back to
        NR_FRAC. This NR_FRAC is the precision with with every operation works throught the iteration steps.

        o. This product has to subtracted from 2. But before this, we need to align the decimal point and also scale 2 with the scaling factor/shift
        So 2 is left shifted by NR_FRAC bits.

        p. Subtract the scaled 2 and d*x_0. The result still has NR_FRAC fractional bits.

        q. Multiply the result from above step with x_0. This will result in  NR_FRAC +  NR_FRAC
        fractional number of bits.

        r. Drop NR_FRAC bits from the above result. Hence the x at each ietration is brought to NR_FRAC bits precision.

        s. Repeat the above steps from m through r for NR_ITER iterations. Then we get the reciprocal of the scaled denominator i.e 1/d'

        t. But we need 1/d, so scale the result from above step with the scale factor. Now we obtain 1/d in fixed point with
        NR_FRAC fractional bits.

        u. To obtain the equivalent floating point number, divide by 2**NR_FRAC

More material for the fixed point divider using Newton Raphson algorithm is available here:
    https://docs.google.com/document/d/16j8StyirLyQ6gH1IHuCZEE-NPAUjVRiK/edit

The script has been modified to cater to the smart RADAR divider where in the numerator is 8Q8 and denominator is 8Q8.
Hence now, I can afford to accomodate bit growth upto the final stage and then drop bits only at the end.

"""

import numpy as np
import sys

np.random.seed(12)

NUM_TOT_BITS = 16
NUM_FRAC_BITS = 8
NUM_SIGN_BIT = 0
NUM_INT_BITS = NUM_TOT_BITS - NUM_FRAC_BITS + NUM_SIGN_BIT

DEN_TOT_BITS = 16
DEN_FRAC_BITS = 8#20#16
DEN_SIGN_BIT = 0
DEN_INT_BITS = DEN_TOT_BITS - DEN_FRAC_BITS + DEN_SIGN_BIT

LUT_LENGTH = 16 # Ideal to have it as a power of 2
LUT_INDEXING_BITWIDTH = int(np.log2(LUT_LENGTH))
ONE_BY_DEN_FRACBITWIDTH = 16#10
NR_FRAC = 22 # Should always be greater than ONE_BY_DEN_FRACBITWIDTH

ONE_BY_DEN_LUT_FLOAT = 1/(np.arange(0.5,1,2**-(LUT_INDEXING_BITWIDTH+1)))
# ONE_BY_DEN_LUT = (np.floor((ONE_BY_DEN_LUT_FLOAT * (2**ONE_BY_DEN_FRACBITWIDTH)) + 0.5)).astype(np.int32)
""" Generate the LUT with high precision (32 bit) and then drop bits to have ONE_BY_DEN_FRACBITWIDTH bit precision for the LUT. This is done to bitmatch RTL divider"""
ONE_BY_DEN_LUT = (np.floor((ONE_BY_DEN_LUT_FLOAT * (2**32)) + 0.5)).astype(np.int64)
ONE_BY_DEN_LUT = ONE_BY_DEN_LUT >> (32-ONE_BY_DEN_FRACBITWIDTH)
""" The 1st entry of the LUT is 1/0.5 = 2. Now, we are representing this 2 with 16 fractional bits. Now, in the RTL implementation,
this LUT is represented with 16 fractional bits and only 1 integer bit i.e 1Q16. So, with 1 integer bit, the max integer value is
1 and hence 2 is not represented/allowed. In python implementation, there is no constraint on the integer part
(as long as the total bitwidth INT + FRAC <= 64).
So, even if the number has an integer part (satisfying the above condition), it can be represented as some xQ16.
Now, to match the RTL simulation, I needed to cap my max value to less than 2 and not allow 2. To take care of this,
The 16 bit fractional representation of 2 is 2 * 2**16 = 2**17 which is an 18 bit number. To make it a 17 bit number,
I have subtracted 1 from 2**17. So, 2**17 - 1 is a 17 bit number of all 1s. Its hexadecimal representation is 1ffff
and in Q format it is 1.ffff

This change has been made to precisely bitmatch RTL implementation as it is and no other logical reason!
"""
ONE_BY_DEN_LUT[0] = ONE_BY_DEN_LUT[0] - 1

A_BY_B_OUTPUT_FRAC_BITS = 16#16

NR_ITER = 3 # Newton-Raphson Iteration


def newton_raphson_divider(num,den,nIter=NR_ITER):

    global LUT_LENGTH, LUT_INDEXING_BITWIDTH, ONE_BY_DEN_FRACBITWIDTH, ONE_BY_DEN_LUT, A_BY_B_OUTPUT_FRAC_BITS


    if (den == 0):
        print('Cannot compute 1/0 !')
        sys.exit()

    temp = den
    count = 0
    while (temp!=0):
        temp = (temp >> 1)
        count += 1

    positionOfLeading1fromLSB = count

    # numLeadingZeros = DEN_TOT_BITS - positionOfLeading1fromLSB
    # scalingFactor = DEN_INT_BITS - numLeadingZeros # If positive, right shift else left shift

    scalingFactor = positionOfLeading1fromLSB - DEN_FRAC_BITS # Scaling factor/shifts can be computed this way. If positive, right shift else left shift

    """ Step k. Extract the index to sample into the LUT"""
    if (positionOfLeading1fromLSB <  (LUT_INDEXING_BITWIDTH+1)):
        indexToSampleLUT = (den << ((LUT_INDEXING_BITWIDTH+1) - positionOfLeading1fromLSB)) & (2**LUT_INDEXING_BITWIDTH - 1)
    else:
        indexToSampleLUT = (den >> (positionOfLeading1fromLSB - (LUT_INDEXING_BITWIDTH+1))) & (2**LUT_INDEXING_BITWIDTH - 1)

    """ Denominator is actually of format DEN_INT_BITS Q DEN_FRAC_BITS but brought to 0 Q NR_FRAC by discarding the
    leading zeros and taking all the bits from the 1st 1 (from the MSB) since this is where the content is.
    Also, this is how the data will be post applying the scaling factor and bringing it to range [0.5,1).
    In this implementation, we want the value to have NR_FRAC bits and hence we left shift to add some bits
    so that the final value is of format 0QNR_FRAC. Note: We are assuming it is brought to range [0.5,1) and hence no integer bits."""
    den = np.int64(den << (NR_FRAC - positionOfLeading1fromLSB))
    x = ONE_BY_DEN_LUT[indexToSampleLUT] # This is the initial 1/d where d is downscaled to lie in [0.5,1)

    for ele in range(nIter):
        """
        x = x * (2 - den*x) # den(scaled denominator) is assumed to be brought to the range [0.5,1)
        """
        if (ele == 0):
            """ For the first iteration only, bring seed from ONE_BY_DEN_FRACBITWIDTH fractional bits to NR_FRAC"""
            x = x << (NR_FRAC-ONE_BY_DEN_FRACBITWIDTH)

        a1 = (den * x) # 0Q(NR_FRAC) * 2Q(NR_FRAC) --> 2Q(NR_FRAC + NR_FRAC)
        a1 = (a1 >> NR_FRAC) # Bring back x*d back to NR_FRAC fractional bits

        """Bring 2 to the same decimal scale as a1"""
        # 2Q0 --> 2Q(NR_FRAC)
        a2 = 2 << (NR_FRAC)


        """ There will be a bit growth post multiplication of den and x and also for the scaling of 2.
        If it is contained to only 32 bit, it will overflow, hence containing the below result in a 64 bit datatype"""
        # x'Q(NR_FRAC) -  2Q(NR_FRAC) --> yQ(NR_FRAC)
        a3 = (a2 - a1).astype(np.int64)

        # 2Q(NR_FRAC) * yQ(NR_FRAC) --> (2+y)Q(NR_FRAC)
        a4 = (x * a3) >> (NR_FRAC)


        x = a4 # fractional bitwidth of x = NR_FRAC
        # print('Iter # = {}'.format(ele))

    """Bring back the scaling factor to the final output to get the true value of 1/d """
    # y'Q(ONE_BY_DEN_FRACBITWIDTH) --> zQ(ONE_BY_DEN_FRACBITWIDTH)
    if (scalingFactor >=0):
        xRescaled = x >> scalingFactor
    else:
        xRescaled = x << np.abs(scalingFactor)

    # NUM_INT_BITSQ(NUM_FRAC_BITS) * zQ(NR_FRAC) --> (NUM_INT_BITS+z)Q(NUM_FRAC_BITS+NR_FRAC)
    numByDen = (num * x) # Dont drop bits here
    """Bring back the scaling factor to the final output to get the true value of n/d. Drop precision post scaling """
    if (scalingFactor >=0):
        numByDenScaled = numByDen >> scalingFactor
    else:
        numByDenScaled = numByDen << np.abs(scalingFactor)

    # (NUM_INT_BITS+z)Q(NUM_FRAC_BITS+NR_FRAC) --> (NUM_INT_BITS+z)Q(A_BY_B_OUTPUT_FRAC_BITS)
    numByDenScaled = numByDenScaled >> (NUM_FRAC_BITS + NR_FRAC - A_BY_B_OUTPUT_FRAC_BITS) #  Drop bits so that final precision is A_BY_B_OUTPUT_FRAC_BITS

    return xRescaled, numByDenScaled


numFloat = np.random.randint(0,2**(NUM_INT_BITS)-1) + np.random.random()#255.99609375#
numFixed = np.floor((numFloat * 2**NUM_FRAC_BITS) + 0.5).astype(np.uint16)

denFloat = np.random.randint(0,2**(DEN_INT_BITS)-1) + np.random.random()#0.05078125# # Failure case for ONE_BY_DEN_FRACBITWIDTH = 16 : 27881.24593943644
denFixed = np.floor((denFloat * 2**DEN_FRAC_BITS) + 0.5).astype(np.uint16)

xRescaled, numByDen = newton_raphson_divider(numFixed,denFixed)

newtonRaphsonBased1byDenFloat = xRescaled/(2**(NR_FRAC))
true1byDen = 1/denFloat

print('Float 1/d = {}'.format(true1byDen))
print('Fixed 1/d = {}'.format(newtonRaphsonBased1byDenFloat))

numByDenFloat = numFloat * true1byDen
numByDenFixedConvertFloat = numByDen/(2**A_BY_B_OUTPUT_FRAC_BITS)

print('Float n/d = {}'.format(numByDenFloat))
print('Fixed n/d = {}'.format(numByDenFixedConvertFloat))

print('Actual div hex n/d = {}'.format(hex(np.int64(np.floor(numByDenFloat*2**A_BY_B_OUTPUT_FRAC_BITS + 0.5)))))
print('Newton div hex n/d = {}'.format(hex(numByDen)))

tvDump = False
if tvDump:
    fh = open(r'D:\Steradian\Git\python_smartradar_stack\rpk\ApodizationIP' + '//divider_tv.txt','w')
    fh.write("\"Num\" \"Den\" \"NumbyDen\" \n")
    count = 0
    while (count < 500000):

        numFloat = np.random.randint(0,2**(NUM_INT_BITS)-1) + np.random.random()#255.99609375#
        numFixed = np.floor((numFloat * 2**NUM_FRAC_BITS) + 0.5).astype(np.uint16)

        denFloat = np.random.randint(0,2**(DEN_INT_BITS)-1) + np.random.random()#0.05078125# # Failure case for ONE_BY_DEN_FRACBITWIDTH = 16 : 27881.24593943644
        denFixed = np.floor((denFloat * 2**DEN_FRAC_BITS) + 0.5).astype(np.uint16)

        if (denFixed == 0):
            print('Cannot compute 1/0 !')
            continue

        xRescaled, numByDen = newton_raphson_divider(numFixed,denFixed)

        numByDen17bit = numByDen & (2**(A_BY_B_OUTPUT_FRAC_BITS+1) -1)

        numFixedHex = hex(numFixed)[2::]
        lennumFixedHex = len(numFixedHex)
        # if (lennumFixedHex < 4):
        numFixedHex = str(0)*(4-lennumFixedHex) + numFixedHex

        denFixedHex = hex(denFixed)[2::]
        lendenFixedHex = len(denFixedHex)
        denFixedHex = str(0)*(4-lendenFixedHex) + denFixedHex

        numByDen17bitHex = hex(numByDen17bit)[2::]
        lennumByDen17bitHex = len(numByDen17bitHex)
        numByDen17bitHex = str(0)*(5-lennumByDen17bitHex) + numByDen17bitHex

        fh.write("{}  {}  {}\n".format(numFixedHex, denFixedHex, numByDen17bitHex))

        count += 1

    fh.close()
