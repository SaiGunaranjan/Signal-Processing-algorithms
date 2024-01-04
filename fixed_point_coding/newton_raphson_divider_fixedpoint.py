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
        Now, the new x i.e. x_i+1 is the location where the line intersects x axis --> y = 0.
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

Will add the fixed point implementation details in the next commit. Also, there is a small bug which causes wrong results
when the d is chosen large! I will fix this issue in the subsequent commits.

"""

import numpy as np
import sys

# np.random.seed(10)

LUT_LENGTH = 32 # Ideal to have it as a power of 2
LUT_INDEXING_BITWIDTH = int(np.log2(LUT_LENGTH))
ONE_BY_DEN_FRACBITWIDTH = 16#10

ONE_BY_DEN_LUT_FLOAT = 1/(np.arange(0.5,1,2**-(LUT_INDEXING_BITWIDTH+1)))
ONE_BY_DEN_LUT = (np.floor((ONE_BY_DEN_LUT_FLOAT * (2**ONE_BY_DEN_FRACBITWIDTH)) + 0.5)).astype(np.int32)

DEN_FRAC_BITS = 16
DEN_INT_BITS = 16
SIGN_BIT = 0
DEN_TOT_BITS = DEN_INT_BITS + DEN_FRAC_BITS - SIGN_BIT


# denFloat = 1.965#100.997652#1.965#15.375
denFloat = np.random.random()*np.random.randint(0,2**(DEN_INT_BITS-1))
denFixed = np.floor((denFloat * 2**DEN_FRAC_BITS) + 0.5).astype(np.int32)

NR_ITER = 3 # Newton-Raphson Iteration

if (denFixed == 0):
    print('Cannot compute 1/0 !')
    sys.exit()

num = denFixed
count = 0
while (num!=0):
    num = (num >> 1)
    count += 1
numLeadingZeros = DEN_TOT_BITS - count
scalingFactor = DEN_INT_BITS - numLeadingZeros # If positive, right shift else left shift

positionOfLeading1fromLSB = count

if (positionOfLeading1fromLSB <  (LUT_INDEXING_BITWIDTH+1)):
    indexToSampleLUT = (denFixed << ((LUT_INDEXING_BITWIDTH+1) - positionOfLeading1fromLSB)) & (2**LUT_INDEXING_BITWIDTH - 1)
else:
    indexToSampleLUT = (denFixed >> (positionOfLeading1fromLSB - (LUT_INDEXING_BITWIDTH+1))) & (2**LUT_INDEXING_BITWIDTH - 1)

denDownScaled = np.int64(denFixed) # Denominator is actually of format DEN_INT_BITS Q DEN_FRAC_BITS but treated as 0 Q scalingFactor+DEN_FRAC_BITS
x = ONE_BY_DEN_LUT[indexToSampleLUT] # This is the initial 1/d where d is downscaled to lie in [0.5,1)


for ele in range(NR_ITER):
    # x = x * (2-(den/scaling)*x)
    # x = x/scaling * ((2*scaling) - (den*x))
    a1 = (denDownScaled * x) #  DEN_FRAC_BITS * 2QONE_BY_DEN_FRACBITWIDTH # Not dropping any bits post d*x
    a2 = 2 << (DEN_FRAC_BITS + ONE_BY_DEN_FRACBITWIDTH) # Bring 2 to the same decimal scale as a1
    if (scalingFactor>=0):
        a2 = a2 << scalingFactor # Multiply 2 (a2) with the scaling factor if scaling factor is +ve (>1)
    else:
        a2 = a2 >> np.abs(scalingFactor) # Divide 2 (a2) with the scaling factor if scaling factor is -ve (<1)
    a3 = (a2 - a1).astype(np.int64)
    a4 = (x * a3) >> (DEN_FRAC_BITS+ONE_BY_DEN_FRACBITWIDTH)
    if  (scalingFactor >=0):
        a4 = a4 >> scalingFactor # Divide x by the scaling factor if scaling factor is +ve (>1)
    else:
        a4 = a4 << np.abs(scalingFactor) # Multiply x by the scaling factor if scaling factor is -ve (<1)
    x = a4 # 2QONE_BY_DEN_FRACBITWIDTH
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


